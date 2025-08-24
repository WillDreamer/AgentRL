#!/usr/bin/env python3
"""
Assume the role passed and send back the token details in a format that the aws-sdk library can use.
Cache those credentials in a file so that we don't call assume role continuously

This is useful when running jobs in Batch queues under a different account than the owner of the
job.
"""

import datetime
import json
import logging
import os
import stat
import sys
import tempfile
import time
from string import ascii_letters, digits
from typing import Any

import boto3

DEFAULT_REGION = "us-east-1"
CREDENTIALS_FILENAME = "credentials"
LOCK_FILENAME = "lock"
DEFAULT_SESSION_NAME = "mfivelib-session"
MINIMUM_TIME_LEFT = datetime.timedelta(minutes=15)
MAX_LOCK_SECONDS = 60 * 10
RFC3339_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def main(args: list[str]) -> int:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        default=0,
        action="count",
        help="Verbosity.  Invoke many times for higher verbosity",
    )
    parser.add_argument("-p", "--assuming-profile", required=True, help="The profile that we'll use to assume the role")
    parser.add_argument("-r", "--region", default=DEFAULT_REGION, help="Region to use (default: %(default)s)")
    parser.add_argument(
        "-s", "--session-name", default=DEFAULT_SESSION_NAME, help="Name to give to the session (default: %(default)s)"
    )
    parser.add_argument("role", nargs=1, help="Role to assume")

    parameters = parser.parse_args(args)

    logging.basicConfig(level=verbosity_to_level(parameters.verbosity))

    profile_name = parameters.assuming_profile
    role = parameters.role[0]
    session_name = parameters.session_name

    credentials, credentials_directory = get_cached_credentials(profile_name, role)
    if not credentials:
        credentials = assume_role(profile_name, role, session_name, parameters.region)
        write_credentials_to_cache(credentials_directory, credentials)
    print(json.dumps(credentials, cls=RFCDatetimeEncoder))

    return 0


def assume_role(profile: str, role: str, session_name: str, region: str) -> dict[Any, Any]:
    session = boto3.Session(profile_name=profile)
    sts = session.client("sts", region_name=region)
    response = sts.assume_role(RoleArn=role, RoleSessionName=session_name)
    sending_back = response["Credentials"]
    # See https://docs.aws.amazon.com/sdkref/latest/guide/feature-process-credentials.html#feature-process-credentials-output
    # for details of what we send back
    sending_back["Version"] = 1
    return sending_back


def get_cached_credentials(profile: str, role: str) -> tuple[dict[Any, Any] | None, str]:
    credentials_directory = get_credentials_directory(profile, role)
    credentials_path = os.path.join(credentials_directory, CREDENTIALS_FILENAME)
    credentials = read_credentials(credentials_path)

    if credentials:
        if datetime.datetime.utcnow() + MINIMUM_TIME_LEFT > credentials["Expiration"]:
            # If they are close to expiring, act as if we don't have any
            credentials = None

    return credentials, credentials_directory


def write_credentials_to_cache(
    credentials_directory: str, credentials: dict[Any, Any], tried_already: bool = False
) -> None:
    """
    Write the credentials to the directory.
    tried_already is a marker to denote that this is the second time that we are trying to
    record them
    """
    credentials_path = os.path.join(credentials_directory, CREDENTIALS_FILENAME)
    lock_path = os.path.join(credentials_directory, LOCK_FILENAME)

    try:
        # opening in exclusive mode. This will fail if it already exists
        with open(lock_path, "x", encoding="utf-8") as lock_file:
            lock_file.write(str(os.getpid()))
    except FileExistsError:
        # Lock already taken. Just in case, check that the lock is valid
        # For validity, we are just going to check if it's less than 10 minutes old
        # since normally this should only be held for sub-second times
        if not tried_already:
            try:
                if os.path.getmtime(lock_path) + MAX_LOCK_SECONDS < time.time():
                    # not valid, wipe it and try again
                    os.remove(lock_path)
                    write_credentials_to_cache(credentials_directory, credentials, tried_already=True)
            except OSError:
                # Maybe the lock file disappeared in between.
                # If that is what happened, leave it as is (it's fresh enough)
                pass
    else:
        try:
            # Write the credentials under a temporary path and then move it to their final name
            # Moving is atomic, so it avoids races with any other process that might be reading
            credentials_file = tempfile.NamedTemporaryFile(
                mode="wt",
                encoding="utf-8",
                dir=credentials_directory,
                prefix="credentials-under-construction",
                delete=False,
            )
            credentials_file.write(json.dumps(credentials, cls=RFCDatetimeEncoder))
            credentials_file.close()
            os.rename(credentials_file.name, credentials_path)
        finally:
            # clean up the lock if we created it
            try:
                os.remove(lock_path)
            except OSError:
                logging.warning("Could not remove %s", lock_path)


def read_credentials(path: str) -> dict[Any, Any] | None:
    try:
        with open(path, encoding="utf-8") as input:
            contents = input.read()
    except OSError:
        # maybe it doesn't exist. This is standard
        return None

    try:
        credentials = json.loads(contents)
    except json.JSONDecodeError:
        # maybe it's half-written
        logging.warning("Error loading json from %s", path)
        return None

    # convert the expiration to a timestamp
    try:
        credentials["Expiration"] = datetime.datetime.strptime(credentials["Expiration"], RFC3339_FORMAT)
    except (KeyError, ValueError):
        # wrong date format or something went wrong
        logging.warning("Error parsing expiration of credentials")
        return None

    return credentials


def get_credentials_directory(profile: str, role: str) -> str:
    """
    Get the directory to use to store the credentials. Should be repeatable.
    Create if it doesn't exist
    """
    normalized_directory_path = replace_non_standard_characters(f"credentials-{profile}-_-{role}")

    try:
        # make it so that only we have access to it
        os.mkdir(normalized_directory_path, 0o700)
    except FileExistsError:
        # this is fine, it just means it was there already. Just make sure that only
        # we have access to it
        if (
            os.path.islink(normalized_directory_path)
            or not os.path.isdir(normalized_directory_path)
            or stat.S_IMODE(os.stat(normalized_directory_path).st_mode) not in [0o700, 0o2700]
        ):
            raise ValueError(  # noqa: B904
                f"{normalized_directory_path} exists but it is either not a directory or it's too permissive"
            )

    return normalized_directory_path


def replace_non_standard_characters(name: str, replacer: str = "_") -> str:
    """
    Sanitize the path
    """
    acceptable = set(ascii_letters + digits + "-_")
    replacement = []
    for character in name:
        if character not in acceptable:
            replacement.append(replacer)
        else:
            replacement.append(character)
    return "".join(replacement)


def verbosity_to_level(verbosity: int) -> int:
    """
    Utility helper to convert a verbosity count to a logging level
    """
    if verbosity == 0:
        return logging.ERROR
    elif verbosity == 1:
        return logging.WARNING
    elif verbosity == 2:
        return logging.INFO
    else:
        return logging.DEBUG


class RFCDatetimeEncoder(json.JSONEncoder):
    """
    Encodes timestamps in a way that AWS likes
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime.datetime):
            # we want something like 2024-03-01T19:32:07Z
            return o.strftime(RFC3339_FORMAT)
        return super().default(o)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

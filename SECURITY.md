# Security Policy

## Reporting a Vulnerability

Please use GitHub's private vulnerability-reporting feature for this
repository. Do not open a public issue containing credentials, private paths,
unreleased dataset information, or exploit details. If private reporting is
not enabled yet, contact a repository maintainer privately and share only the
minimum information needed to establish a secure follow-up channel.

## Credential Policy

- Never store API keys, access tokens, passwords, private keys, or cloud
  credentials in source files or committed configuration.
- Use environment variables for secrets. `.env` files are local-only and are
  ignored; `.env.example` may contain variable names but never values.
- W&B must remain optional. Public inference must not require a W&B account.
- Do not place secrets in command-line examples, logs, checkpoints, notebooks,
  generated metadata, issue reports, or experiment snapshots.
- Never assume that deleting a secret from the current branch removes it from
  Git history.

## Private Infrastructure Policy

Public code and configs must not contain personal home directories, cluster
hostnames, scheduler account names, private mount points, internal URLs, or
nonpublic dataset locations. Paths must be relative, user-configurable, or
provided through environment variables and CLI options.

## Response to an Exposed Secret

1. Revoke or rotate the credential at its provider immediately.
2. Review provider audit logs and remove unauthorized sessions or grants.
3. Remove the credential from the working tree and all published artifacts.
4. Decide whether repository history must be rewritten. Coordinate this before
   force-pushing because history rewriting disrupts collaborators and forks.
5. Run a history-aware secret scanner and verify that the old credential no
   longer works.
6. Document the incident privately, including its exposure window and the
   remediation performed.

## Release Checks

Run the following before each public push or release:

```bash
bash scripts/check_release_hygiene.sh
```

The automated check is a guardrail, not proof that a release is safe. Review
new configs, examples, checkpoints, metadata, archives, and Git history
manually as well.

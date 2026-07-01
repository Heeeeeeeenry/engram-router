# Security Policy

EngramRouter handles private memory. Treat all stored data as sensitive.

## Security Principles

- Local-first by default.
- No default cloud sync.
- No hidden telemetry.
- No silent deletion.
- No inferred facts disguised as user-stated facts.

## Reporting Issues

Do not publish private user memory examples in public issues. Redact all personal data.

## Data Handling

The first implementation stores data locally in SQLite. Users should be able to inspect, back up and delete their own database.

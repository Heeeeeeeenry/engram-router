# Contributing to EngramRouter

Thanks for helping build EngramRouter.

## Before Changing Anything

Read these first:

1. `docs/PROJECT_BRIEF.md`
2. `docs/DEVELOPMENT_RULES.md`
3. `docs/ROADMAP.md`

## The Most Important Rule

Do not turn EngramRouter into a generic RAG project.

The project exists to replace lossy summary compression with evidence-based, on-demand memory recall for agents.

## Required For Every Feature

- Add or update tests.
- Preserve raw evidence.
- Respect top-k/minimal-context behavior.
- Update docs when behavior changes.

## Pull Request Checklist

- [ ] Raw memory evidence is preserved.
- [ ] Irrelevant memory is not injected by default.
- [ ] Missing information leads to gap output or clarification.
- [ ] New behavior is documented.
- [ ] Tests pass.

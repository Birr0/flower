See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Setting up a development environment manually

You can set up a development environment using a venv by running:

```zsh
python3 -m venv venv          # create a virtualenv called venv
source ./venv/bin/activate   # now `python` points to the virtualenv python
pip install -v -e ".[dev]"    # -v for verbose, -e for editable, [dev] for dev dependencies
```

Alternatively, you can create a conda environment, activate it and then `pip install -v -e ".[dev]"` as described above.

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # this will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.


# Contributing to the repo

Each commit should use one of the following prefixes:

    | Prefix     | Use when                                |
    |------------|-----------------------------------------|
    | [Feature]  | New functionality                       |
    | [Fix]      | Bug fix (simple, not urgent)            |
    | [Hotfix]   | Urgent production fix                   |
    | [Refactor] | Restructuring code, no behavior change  |
    | [Style]    | Formatting, linting, whitespace         |
    | [Docs]     | Documentation changes                   |
    | [Test]     | Adding or modifying tests               |
    | [Chore]    | Dependency updates, CI, config, tooling |
    | [Perf]     | Performance improvements                |
    | [Break]    | Breaking API / behavior change          |

    A few that are situational:

    | Prefix   | Use when                                  |
    |----------|-------------------------------------------|
    | [Revert] | Reverting a previous commit               |
    | [WIP]    | Work in progress (pre-PR)                 |
    | [Config] | Environment, server, or infra config      |
    | [Rename] | Files, classes, or variables renamed only |

    Full message format suggestion:


    [Prefix] short description

    Optional longer explanation in body if the commit isn't self-explanatory.
    Keep the subject line under 72 characters.


    Examples:

    [Feature] add user profile avatar upload

    [Refactor] extract validation logic from signup handler

    [Fix] handle null timezone in user dashboard

    [Style] apply formatter to auth module


    A few principles to keep it maintainable:

    1. Don't over-split. If a commit touches [Feature] and [Style], pick the one that matters more. Usually the feature — style changes will be obvious in the diff.
    2. [Style] is for formatting-only changes. If there's real code in there, it's a [Refactor] or [Fix] that happens to have reformatting too.
    3. [Chore] is your catch-all. Dep bumps, GitHub Actions updates, env vars — anything that doesn't change application behavior.
    4. Skip the prefix for merge commits or trivial things (fixing a typo in a comment, etc.). The prefix system is for commits with real intent.
    5. Stick to your set. When you feel the urge to create [X], ask whether an existing prefix covers it. If three or more people independently reach for a new one, then add it.

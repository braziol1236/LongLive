
## LongLive OSS Contribution Rules

#### Issue Tracking

* All enhancement, bugfix, or change requests must begin with the creation of a [LongLive Issue Request](https://github.com/nvidia/LongLive/issues).
  * The issue request must be reviewed by LongLive engineers and approved prior to code review.


#### Coding Guidelines

- All source code contributions must strictly adhere to the [LongLive Coding Guidelines](CODING-GUIDELINES.md).

- In addition, please follow the existing conventions in the relevant file, submodule, module, and project when you add new code or when you extend/fix existing functionality.

- To maintain consistency in code formatting and style, you should also run `clang-format` on the modified sources with the provided configuration file. This applies LongLive code formatting rules to:
  - class, function/method, and variable/field naming
  - comment style
  - indentation
  - line length

- Format git changes:
  ```bash
  # Commit ID is optional - if unspecified, run format on staged changes.
  git-clang-format --style file [commit ID/reference]
  ```

- Format  individual source files:
  ```bash
  # -style=file : Obtain the formatting rules from .clang-format
  # -i : In-place modification of the processed file
  clang-format -style=file -i -fallback-style=none <file(s) to process>
  ```

- Format entire codebase (for project maintainers only):
  ```bash
  find samples plugin -iname *.h -o -iname *.c -o -iname *.cpp -o -iname *.hpp \
  | xargs clang-format -style=file -i -fallback-style=none
  ```

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:
  - Avoid committing commented-out code.
  - Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description. The more complex the changes are in a single PR, the more time it will take to review those changes.

- Write commit titles using imperative mood and [these rules](https://chris.beams.io/posts/git-commit/), and reference the Issue number corresponding to the PR. Following is the recommended format for commit texts:
```
#<Issue Number> - <Commit Title>

<Commit Body>
```

- Ensure that the build log is clean, meaning no warnings or errors should be present.

- Ensure that all `sample_*` tests pass prior to submitting your code.

- All OSS components must contain accompanying documentation (READMEs) describing the functionality, dependencies, and known issues.

  - See `README.md` for existing samples and plugins for reference.

- All OSS components must have an accompanying test.

  - If introducing a new component, such as a plugin, provide a test sample to verify the functionality.

- To add or disable functionality:
  - Add a CMake option with a default value that 

---

> **Personal note (fork):** For my own workflow I run clang-format as a pre-commit hook so I never forget it.
> Quick setup: `cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`
> (The `scripts/pre-commit` hook just calls the `git-clang-format` command listed above.)

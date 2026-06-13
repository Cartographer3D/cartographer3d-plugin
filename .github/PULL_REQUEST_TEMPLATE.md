<!-- PR title: use `fix:`, `feat:`, `chore:`, or `perf:` so auto-labeling and release notes work. Use `!` for breaking changes. -->

## Reason / issue reference

<!-- Why is this change needed? Link or close the issue if there is one. -->

Closes #

## Functional changes

<!-- What user-facing behavior changed? Focus on what users can do now, what behaves differently, or what problem is resolved. Avoid summarizing code changes; reviewers can see implementation details in the diff. -->

## Decisions and callouts

<!-- Design choices, tradeoffs, compatibility concerns, or specific areas where reviewer attention would help.
Remove this section if there is nothing to call out. -->

## Manual testing

<!-- Optional. Include hardware/firmware target, commands/macros run, or observed results.
Remove this section if not applicable. -->

## Install

```bash
~/klippy-env/bin/pip install --no-cache-dir --force-reinstall "git+https://github.com/Cartographer3D/cartographer3d-plugin.git@BRANCH_OR_SHA"
sudo systemctl restart klipper
```

To revert to the latest stable release:

```bash
~/klippy-env/bin/pip install --no-cache-dir --force-reinstall cartographer3d-plugin
sudo systemctl restart klipper
```

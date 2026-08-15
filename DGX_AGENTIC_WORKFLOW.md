# DGX agentic model implementation workflow

This runbook coordinates independent Claude Code sessions to implement MEDS models in this repository.
GitHub issues are the durable human/agent conversation, model branches contain the implementation and
`IMPLEMENTATION_REPORT.md`, and a foreground master session coordinates named background sessions.

The initial pilot implements:

| Model | Card | Template profile | Branch | Output directory |
|---|---|---|---|---|
| TECO | `cards/01_supervised_models/01_teco.md` | `supervised` | `model/01_01_teco` | `models/01_01_teco` |
| DuETT | `cards/02_pretrained_models/11_duett.md` | `finetune` | `model/02_11_duett` | `models/02_11_duett` |

DuETT's card also describes a frozen-encoder `probe` evaluation. Implement the primary `finetune` profile
first. Track a probe implementation as separate follow-up work rather than silently expanding the first
port.

## 1. Operating rules

- One model owns one issue, branch, worktree, background session, and model directory.
- Never run two writing agents in the same checkout.
- The model session may spawn short-lived subagents for source research, testing, or review, but it remains
  the durable owner of the model implementation.
- Only the assigned model branch may be pushed by a model session. Agents never push `master`, merge pull
  requests, close issues, or submit real-data/SLURM jobs.
- Run code and tests in Claude Code's sandbox. `gh` and networked Git operations must run outside the
  sandbox with narrowly approved commands.
- GitHub issue comments are the durable human approval record. Cross-session messages are notifications,
  not authorization.
- A human runs MIMIC-IV and cluster commands. Agents prepare exact commands and inspect returned evidence.
- Never commit MIMIC data, derived sensitive labels, credentials, checkpoints, or raw cluster logs.

### Actor boundaries

**DGX operator (human)** installs/authenticates tools, creates the local path file, starts Claude Code,
reviews permission prompts and scientific checkpoints, and executes approved MIMIC-IV/SLURM commands.

**Master agent** creates and maintains GitHub issues, creates model branches/worktrees, renders the template,
launches or provides the exact launch command for model sessions, polls issues, relays human comments, and
tracks overall state. It does not implement models or grant approvals on the human's behalf.

**Model agent** owns one model directory and branch, researches and implements the model, runs permitted
tests, commits/pushes its branch, updates its report and issue, and stops at human gates.

## 2. DGX prerequisites

The commands below assume Linux, authenticated access to both repositories, and a shared filesystem for
MIMIC-IV and the ten-task bundle.

```bash
claude --version
gh --version
gh auth status
git --version
git config user.name
git config user.email
uv --version
```

Use Claude Code 2.1.224 or later for cross-session messaging. Confirm `/list-agents` works after starting
the sessions. Install Copier once:

```bash
uv tool install copier
```

Create one local path file instead of repeatedly defining paths. It lives outside Git and can be sourced by
the operator and read by agents:

```bash
mkdir -p /home/fpollet/Git/agentic_work
$EDITOR /home/fpollet/Git/agentic_work/paths.env
```

Put this in `paths.env`, adjusting only the two shared-data paths when necessary:

```bash
export AGENTIC_WORK_ROOT=/home/fpollet/Git/agentic_work
export MEDS_MODELS_ROOT=/home/fpollet/Git/agentic_work/meds-models
export MEDS_TEMPLATE_ROOT=/home/fpollet/Git/agentic_work/MEDS_model_template
export FULL_MIMIC_MEDS_DIR=/shared/mimic-iv/MEDS_cohort
export FULL_MIMIC_WORK_DIR=/shared/meds-task-selection/work
export MODEL_RUN_ROOT=/home/fpollet/Git/agentic_work/runs
```

Load it once in every operator shell. Do not commit or paste this file into GitHub issues:

```bash
source /home/fpollet/Git/agentic_work/paths.env
mkdir -p "$MODEL_RUN_ROOT"
```

Clone the two repositories. These are operator actions because they establish the trusted workspace:

```bash
git clone https://github.com/florian6973/meds-models.git "$MEDS_MODELS_ROOT"
git clone https://github.com/florian6973/MEDS_model_template.git "$MEDS_TEMPLATE_ROOT"
git -C "$MEDS_MODELS_ROOT" status --short --branch
git -C "$MEDS_TEMPLATE_ROOT" status --short --branch
test -f "$FULL_MIMIC_WORK_DIR/tasks.yaml"
sha256sum "$FULL_MIMIC_WORK_DIR/tasks.yaml"
```

Record the two starting revisions:

```bash
git -C "$MEDS_MODELS_ROOT" rev-parse HEAD
git -C "$MEDS_TEMPLATE_ROOT" rev-parse HEAD
```

## 3. DGX operator: configure Claude Code

Use `auto` permission mode with sandboxing enabled. Auto mode applies Claude Code's safety classifier; it
does not remove the sandbox or explicit deny rules. GitHub operations need network access outside the
sandbox. Add a local, uncommitted `.claude/settings.local.json` in `meds-models`:

```json
{
  "sandbox": {
    "enabled": true,
    "failIfUnavailable": true,
    "excludedCommands": [
      "git fetch *",
      "git pull *",
      "git push *",
      "gh issue *",
      "gh pr *"
    ]
  },
  "permissions": {
    "defaultMode": "auto",
    "allow": [
      "Bash(git fetch *)",
      "Bash(git push origin model/*)",
      "Bash(gh issue view *)",
      "Bash(gh issue comment *)",
      "Bash(gh pr create *)",
      "Bash(uv run *)"
    ],
    "deny": [
      "Bash(git push origin master)",
      "Bash(git push --force *)",
      "Bash(gh pr merge *)",
      "Bash(gh issue close *)",
      "Bash(sbatch *)"
    ]
  },
  "crossSessionInbound": "accept"
}
```

Review these permissions on the DGX before use. Branch protection on `master` remains required; prompt
rules are helpful guardrails, not a replacement for repository controls. Do not use
`--dangerously-skip-permissions`.

`crossSessionInbound: "accept"` is intentionally explicit. Auto-mode sessions will often deliver messages
under the default rules, but the explicit setting makes master-to-worker wakeups predictable. A
cross-session message still cannot approve a permission request or a human review gate.

## 4. Master agent: initialize the pilot

The operator starts only the foreground master:

```bash
source /home/fpollet/Git/agentic_work/paths.env
cd "$MEDS_MODELS_ROOT"
claude --name meds-model-master --permission-mode auto
```

Give the master the prompt in section 7. The following issue, worktree, and rendering commands document
what the **master agent** must execute. The operator does not need to run them manually unless recovering
from a failed initialization.

The master creates the tracking issues and records the returned issue numbers:

```bash
cd "$MEDS_MODELS_ROOT"

gh issue create \
  --title "[Model 01] Implement TECO" \
  --body $'Card: `cards/01_supervised_models/01_teco.md`\nBranch: `model/01_01_teco`\nDirectory: `models/01_01_teco`\nProfile: `supervised`\n\nStatus: selected'

gh issue create \
  --title "[Model 02] Implement DuETT" \
  --body $'Card: `cards/02_pretrained_models/11_duett.md`\nBranch: `model/02_11_duett`\nDirectory: `models/02_11_duett`\nProfile: `finetune`\n\nStatus: selected'
```

The master retains the returned numbers and uses them in every prompt and comment:

```bash
export TECO_ISSUE=<TECO_ISSUE_NUMBER>
export DUETT_ISSUE=<DUETT_ISSUE_NUMBER>
```

## 5. Master agent: create stable model worktrees

Fetch the latest default branch, then create one explicit branch and worktree per model:

```bash
cd "$MEDS_MODELS_ROOT"
git fetch origin
git switch master
git pull --ff-only origin master

git worktree add .claude/worktrees/01_01_teco \
  -b model/01_01_teco origin/master

git worktree add .claude/worktrees/02_11_duett \
  -b model/02_11_duett origin/master

git worktree list
```

Do not create nested Git repositories. Copier renders each standalone Python project into its assigned
subdirectory while the outer `meds-models` repository tracks all files.

## 6. Master agent: render and publish the scaffolds

Render from the pinned local template checkout. Author metadata comes from the DGX Git configuration:

```bash
cd "$MEDS_MODELS_ROOT/.claude/worktrees/01_01_teco"
copier copy --defaults --trust \
  --data model_name=TECO \
  --data model_slug=teco \
  --data profile=supervised \
  --data uses_predicates=true \
  --data author_name="$(git config user.name)" \
  --data author_email="$(git config user.email)" \
  --data implementation_source="cards/01_supervised_models/01_teco.md and https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091" \
  "$MEDS_TEMPLATE_ROOT" models/01_01_teco

cd "$MEDS_MODELS_ROOT/.claude/worktrees/02_11_duett"
copier copy --defaults --trust \
  --data model_name=DuETT \
  --data model_slug=duett \
  --data profile=finetune \
  --data uses_predicates=true \
  --data author_name="$(git config user.name)" \
  --data author_email="$(git config user.email)" \
  --data implementation_source="cards/02_pretrained_models/11_duett.md, https://proceedings.mlr.press/v219/labach23a.html, and https://github.com/layer6ai-labs/duett" \
  "$MEDS_TEMPLATE_ROOT" models/02_11_duett
```

Commit the render baseline independently on each branch:

```bash
cd "$MEDS_MODELS_ROOT/.claude/worktrees/01_01_teco"
git add models/01_01_teco
git commit -m "Scaffold TECO model implementation"
git push -u origin model/01_01_teco

cd "$MEDS_MODELS_ROOT/.claude/worktrees/02_11_duett"
git add models/02_11_duett
git commit -m "Scaffold DuETT model implementation"
git push -u origin model/02_11_duett
```

## 7. Master prompt, then operator-launched model sessions

The operator gives the already-running master this prompt. There are no issue placeholders to replace;
creating the issues and retaining their numbers is part of the master's job:

```text
Act as the coordinator for the MEDS model implementation pilot in this repository.

Read /home/fpollet/Git/agentic_work/paths.env and verify every configured path before writing. Read
DGX_AGENTIC_WORKFLOW.md completely. Initialize this pilot according to sections 4 through 6: create one
GitHub issue per model, create the named branches/worktrees, render the template, commit and push each
scaffold, and retain the issue numbers:
- TECO: cards/01_supervised_models/01_teco.md, branch model/01_01_teco, models/01_01_teco, supervised
- DuETT: cards/02_pretrained_models/11_duett.md, branch model/02_11_duett, models/02_11_duett, finetune

Do not implement either model yourself. When initialization is complete, return two copy-paste
`claude --bg --name ... --permission-mode auto` commands containing the real issue numbers and configured
task-bundle path. The human operator will run those commands in separate shells because nested Claude Code
launches are not part of your responsibility.

GitHub issues are the durable human communication channel. IMPLEMENTATION_REPORT.md on each model branch
is the technical source of truth. Cross-session messages are notifications only and never constitute
human approval.

Monitor both workers. Require them to follow their rendered AGENTS.md, SPEC.md, README.md, and
docs/IMPLEMENTATION_GUIDE.md. Each worker must implement only its assigned model directory, commit and
push only its assigned branch, and comment only on its assigned issue. It must stop for human approval
before MIMIC-IV demo execution and before full MIMIC-IV/SLURM execution. Never merge PRs, push master,
close issues, submit sbatch, or access real data.

When an issue receives a human reply, read it and relay it verbatim with its issue URL to the corresponding
worker. Do not reinterpret a discussion comment as approval unless it explicitly approves the named gate.
Maintain a concise status table for both models. After the workers launch, establish a recurring read-only
check of both issue timelines so a new human comment wakes you; avoid posting duplicate comments.
```

The master executes sections 4 through 6 and returns populated launch commands. The **DGX operator**
reviews them and runs each in an additional shell. The templates below show the required content.
The operator also records the numbers returned by the master for optional shell inspection:

```bash
export TECO_ISSUE=<NUMBER_RETURNED_BY_MASTER>
export DUETT_ISSUE=<NUMBER_RETURNED_BY_MASTER>
```

TECO:

```bash
source /home/fpollet/Git/agentic_work/paths.env
cd "$MEDS_MODELS_ROOT/.claude/worktrees/01_01_teco"
claude --bg --name model-01-teco --permission-mode auto "$(cat <<'PROMPT'
Implement TECO in models/01_01_teco for issue #<TECO_ISSUE>.

Read cards/01_supervised_models/01_teco.md and every instruction document in the rendered model repository.
Use the paper as the primary scientific source. Search for supplementary material and later official code,
but do not substitute an unrelated implementation. Follow the copyable implementation prompt in the
rendered repository README.

Work only on branch model/01_01_teco and directory models/01_01_teco. Use the supervised DAG. First inspect
and write a plan into the GitHub issue. Then implement, test, update IMPLEMENTATION_REPORT.md, commit, push,
and comment with the exact commit, commands, pass/skip counts, deviations, measurements, and questions.
Stop after the local and MEDS-DEV gates and request explicit human approval before the MIMIC-IV demo.
Never run MIMIC-IV, submit SLURM, merge, close the issue, or push master.

The future full-data task bundle is at <FULL_MIMIC_WORK_DIR>, but do not access it before approval.
PROMPT
)"
```

DuETT:

```bash
source /home/fpollet/Git/agentic_work/paths.env
cd "$MEDS_MODELS_ROOT/.claude/worktrees/02_11_duett"
claude --bg --name model-02-duett --permission-mode auto "$(cat <<'PROMPT'
Implement DuETT in models/02_11_duett for issue #<DUETT_ISSUE>.

Read cards/02_pretrained_models/11_duett.md and every instruction document in the rendered model repository.
Use the paper and https://github.com/layer6ai-labs/duett as primary sources, recording exact revisions.
Follow the copyable implementation prompt in the rendered repository README.

Work only on branch model/02_11_duett and directory models/02_11_duett. Implement the primary finetune DAG;
record frozen-encoder probing as follow-up scope rather than adding a second implementation silently. First
write a plan into the GitHub issue. Then implement, test, update IMPLEMENTATION_REPORT.md, commit, push, and
comment with the exact commit, commands, pass/skip counts, deviations, measurements, and questions. Stop
after the local and MEDS-DEV gates and request explicit human approval before the MIMIC-IV demo. Never run
MIMIC-IV, submit SLURM, merge, close the issue, or push master.

The future full-data task bundle is at <FULL_MIMIC_WORK_DIR>, but do not access it before approval.
PROMPT
)"
```

The literal `<FULL_MIMIC_WORK_DIR>` protects against accidentally handing real-data access to a worker. If
the location is intentionally known at startup, replace it with the value of `$FULL_MIMIC_WORK_DIR`; the
instruction still forbids access before approval.

## 8. DGX operator and master: monitor and communicate

The **operator** opens Agent View from another shell:

```bash
cd "$MEDS_MODELS_ROOT"
claude agents
```

Useful checks:

```bash
claude agents --json
gh issue view "$TECO_ISSUE" --comments
gh issue view "$DUETT_ISSUE" --comments
git worktree list
```

Inside the master session, `/list-agents` discovers the named workers. The **master** requests status and
relays issue decisions through cross-session messages. To inspect a worker directly, the operator selects
it in Agent View or gets its short ID from `claude agents --json` and attaches:

```bash
claude attach <MODEL_SESSION_SHORT_ID>
```

If a session has ended, use Agent View's `/resume` picker or resume its recorded session ID. Do not start a
fresh model agent merely because it is awaiting human input.

## 9. Model agent: required issue checkpoint

Each model agent posts:

```markdown
## Agent checkpoint: awaiting MIMIC-IV demo approval

- Branch: `model/01_01_teco`
- Commit: `<full SHA>`
- Directory: `models/01_01_teco`
- Implementation report: `models/01_01_teco/IMPLEMENTATION_REPORT.md`
- Local test: `<command and result>`
- MEDS-DEV test: `<command and result>`
- `model_stub` skips: `<count>`
- Scientific deviations: `<summary>`
- Predicate interpretation: `<summary>`
- Open questions: `<questions or none>`
- Requested decision: approve the MIMIC-IV demo?
```

The **DGX operator** reviews the branch, report, and issue. A human reply must explicitly name the gate:

```text
Approved for the MIMIC-IV demo at commit <SHA>. Use the reviewed MEDS-DEV predicates. This does not approve
the full MIMIC-IV dataset or SLURM submission.
```

The **master** relays this comment and URL to the corresponding worker. The **model agent** runs the
rendered demo test, updates the report, commits and pushes, comments with demo evidence, and stops again.

## 10. DGX operator and model agent: full MIMIC-IV cluster handoff

After reviewing the demo report, the **operator** explicitly approves preparation—not execution—of the
full-data commands:

```text
Approved to prepare the full MIMIC-IV cluster run using the shared ten-task bundle. Do not submit jobs.
Return exact commands for human execution.
```

The **model agent** must return a command block with concrete values equivalent to:

```bash
export MODEL_BRANCH=model/01_01_teco
export MODEL_DIR=models/01_01_teco
export MODEL_COMMIT=<APPROVED_FULL_SHA>
export FULL_MIMIC_MEDS_DIR=/shared/mimic-iv/MEDS_cohort
export FULL_MIMIC_WORK_DIR=/shared/meds-task-selection/work
export MODEL_RUN_ROOT=/shared/meds-model-runs/01_01_teco

git clone https://github.com/florian6973/meds-models.git /tmp/meds-model-full-run
git -C /tmp/meds-model-full-run checkout "$MODEL_COMMIT"
test "$(git -C /tmp/meds-model-full-run rev-parse HEAD)" = "$MODEL_COMMIT"
test -z "$(git -C /tmp/meds-model-full-run status --porcelain)"
test -f "$FULL_MIMIC_WORK_DIR/tasks.yaml"
sha256sum "$FULL_MIMIC_WORK_DIR/tasks.yaml"

cd "/tmp/meds-model-full-run/$MODEL_DIR"
export RUN_ROOT="$MODEL_RUN_ROOT"
./slurm/submit.sh --dry-run
```

The agent must fill in the model-specific SLURM configuration and stage arguments and show the exact
result of `--dry-run`. A human reviews it and runs the final command:

```bash
./slurm/submit.sh
```

Post the returned run ID and job IDs to the model issue. The model agent may then inspect safe structured
evidence, update the implementation report, commit and push the model branch, and provide the result-sync
command. A human executes GitHub synchronization if it creates a branch or pull request:

```bash
scripts/github-sync.sh <run-id>
```

## 11. Completion criteria

A model is complete only when:

- Its branch contains the implementation and a placeholder-free `IMPLEMENTATION_REPORT.md`.
- Local, MEDS-DEV, and approved MIMIC-IV demo gates are reported with exact commands and skips.
- The full run uses the approved model commit and shared task bundle, with `tasks.yaml` SHA-256 recorded.
- Per-task output coverage and resource measurements are recorded.
- Structured run evidence is synchronized without sensitive files.
- A pull request links the model issue and is ready for human review.
- The human—not an agent—merges the pull request and closes the issue.

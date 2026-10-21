# Session continuity

On the first turn of a new session in this project, read `HANDOFF.md` if it
exists and give a brief recap of the last state, unfinished work or blockers,
and the next step. Do this once per session, then address the current request.
Treat the handoff as historical context; current user instructions take priority.
Verify relevant state before acting on old commands. Do not launch saved jobs
merely because they appear in the summary. If no handoff exists, continue normally.

A message containing `/handoff` means to use the handoff skill to save the
session. This is a conversational alias, not a registered native slash command.
`$handoff` is the explicit Codex skill invocation.

@echo off
for %%i in (6 7 10 11 13) do (
    start "" julia --project=. -e "cd(\"results/DG2shade\"); ARGS = [\"DG2shade\", \"5\", \"%%i\"]; include(\"../../test/run.jl\")"
)
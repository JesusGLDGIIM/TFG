@echo off
for %%i in (6 7 10 11 13) do (
    start "" julia --project=. -e "cd(\"results/DG2shadeils\"); ARGS = [\"DG2shadeils\", \"5\", \"%%i\"]; include(\"../../test/run.jl\")"
)

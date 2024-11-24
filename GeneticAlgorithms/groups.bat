@echo off
for /L %%i in (11,1,13) do (
    start "" cmd /k julia --project=. -e "cd(\"results/groups\"); ARGS = [\"Grouping\", \"1\", \"%%i\"]; include(\"../../test/run.jl\")"
)
pause


@echo off
for /L %%i in (1,1,15) do (
    start "" julia --project=. -e "cd(\"results/ERDGshadeils\"); ARGS = [\"ERDGshadeils\", \"5\", \"%%i\"]; include(\"../../test/run.jl\")"
)

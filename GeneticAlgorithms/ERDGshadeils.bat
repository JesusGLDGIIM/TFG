@echo off
for /L %%i in (1,1,15) do (
    start "" julia --project=. -e "cd(\"results/ERDGshadeils\"); ARGS = [\"ERDGshadeils\", \"25\", \"%%i\"]; include(\"../../test/run.jl\")"
)

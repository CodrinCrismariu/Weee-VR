// The game core's self-test on Linux (no phone): physics against the ITTF ball spec, the paddle, the
// opponent's aim and a whole exchange. Run scripts/game-test/run.sh inside WSL.
for line in GameSelfTest.run() {
    print(line)
}

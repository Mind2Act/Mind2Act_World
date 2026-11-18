# Environment

`sim/` owns Isaac startup, clocks, cameras and asset resolution. `robot/` owns
the shared X5 robot and appearance/contact contracts. Task code stays in `task/`.
Asset binaries are downloaded separately and verified before use.

from app.core.crash_log import install

install("desktop")

from app.gui.main import main

raise SystemExit(main())

# Use GTK+ instead of GTK+3 for X11 forwarding
PACKAGECONFIG[gtk] = "-DWITH_GTK=ON,-DWITH_GTK=OFF,gtk+,"

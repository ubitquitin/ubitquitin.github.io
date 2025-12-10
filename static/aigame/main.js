const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow () {
    // Create the browser window.
    const win = new BrowserWindow({
        width: 920,  // Slightly larger than your canvas (900)
        height: 690, // Slightly larger than your canvas (650)
        resizable: false, // Keep it fixed like a game console
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        // Optional: Remove the top menu bar for a cleaner game look
        autoHideMenuBar: true 
    });

    // Load your game file
    win.loadFile('index.html');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
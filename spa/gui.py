import subprocess
import tkinter as tk
from tkinter import messagebox

# Initialize the process variable
process = None

# Function to run navigation.py
def start_navigation_script():
    global process
    if process is None:
        try:
            # Run navigation.py as a separate process
            process = subprocess.Popen(["python", "c.py"])
            messagebox.showinfo("Info", "Navigation script is running.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start the navigation script: {e}")
    else:
        messagebox.showinfo("Info", "Navigation script is already running.")

# Function to stop navigation.py
def stop_navigation_script():
    global process
    if process:
        process.terminate()  # Terminate the process
        process = None
        messagebox.showinfo("Info", "Navigation script has been stopped.")
    else:
        messagebox.showinfo("Info", "Navigation script is not running.")

# Set up the GUI
root = tk.Tk()
root.title("Nose Controlled Mouse Navigation")

# Create Start and Stop buttons
start_button = tk.Button(root, text="Start Navigation", command=start_navigation_script, font=("Arial", 16))
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop  Navigation", command=stop_navigation_script, font=("Arial", 16))
stop_button.pack(pady=10)

# Start the GUI main loop
root.mainloop()

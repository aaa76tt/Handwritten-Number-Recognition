"""
Handwritten number Recognition Application - 28x28 Canvas Version
Includes recognition, clear, and training buttons
"""
import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import threading


# ==================== Configuration ====================
CANVAS_SIZE = 28
PIXEL_SIZE = 15
MODEL_PATH = "number_model.pth"

# Dark theme colors
DARK_BG = "#1e1e1e"
DARK_FG = "#ffffff"
CANVAS_BG = "#2d2d2d"
CANVAS_DRAW = "#ffffff"
BUTTON_BG = "#3c3c3c"
BUTTON_FG = "#ffffff"
BUTTON_ACTIVE_BG = "#505050"


# ==================== Neural Network Model ====================
class numberRecognitionNet(nn.Module):
    """Two hidden layer fully connected neural network"""
    
    def __init__(self):
        super(numberRecognitionNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)   # First layer: 784 -> 256
        self.fc2 = nn.Linear(256, 128)   # Second layer: 256 -> 128
        self.fc3 = nn.Linear(128, 10)    # Output layer: 128 -> 10
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
    
    def predict(self, image_data):
        """Predict a single image"""
        if image_data.shape == (28, 28):
            image_data = image_data.flatten()
        
        x = torch.FloatTensor(image_data).unsqueeze(0)
        self.eval()
        
        with torch.no_grad():
            output = self.forward(x)
            probabilities = output[0].numpy()
            predicted_number = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_number])
        
        return predicted_number, confidence


# ==================== Drawing Canvas Component ====================
class DrawingCanvas:
    """28x28 pixel drawing canvas"""
    
    def __init__(self, parent):
        self.size = CANVAS_SIZE
        self.pixel_size = PIXEL_SIZE
        self.drawing = False
        
        # Initialize 28x28 data
        self.data = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Create Canvas
        canvas_width = self.size * self.pixel_size
        canvas_height = self.size * self.pixel_size
        
        self.canvas = tk.Canvas(
            parent,
            width=canvas_width,
            height=canvas_height,
            bg=CANVAS_BG,
            highlightthickness=1,
            highlightbackground=CANVAS_DRAW
        )
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # Initialize rectangles
        self.rectangles = {}
        self._init_rectangles()
        
        self.on_change_callback = None
    
    def _init_rectangles(self):
        """Initialize all pixel rectangles"""
        for i in range(self.size):
            for j in range(self.size):
                x1 = j * self.pixel_size
                y1 = i * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                
                rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=CANVAS_BG,
                    outline=CANVAS_BG
                )
                self.rectangles[(i, j)] = rect_id
    
    def on_mouse_press(self, event):
        self.drawing = True
        self._draw_at_position(event.x, event.y)
    
    def on_mouse_drag(self, event):
        if self.drawing:
            self._draw_at_position(event.x, event.y)
    
    def on_mouse_release(self, event):
        self.drawing = False
    
    def _draw_at_position(self, x, y):
        """Draw at specified position with semi-transparent border effect"""
        col = int(x / self.pixel_size)
        row = int(y / self.pixel_size)
        
        if 0 <= row < self.size and 0 <= col < self.size:
            # Fill center point completely
            self.data[row, col] = 1.0
            rect_id = self.rectangles[(row, col)]
            self.canvas.itemconfig(rect_id, fill=CANVAS_DRAW, outline=CANVAS_DRAW)
            
            # Add semi-transparent border within radius 2
            radius = 2
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    # Skip center point (already processed)
                    if dr == 0 and dc == 0:
                        continue
                    
                    # Calculate distance
                    distance = (dr**2 + dc**2) ** 0.5
                    
                    # Only process points within radius 2
                    if distance <= radius:
                        new_row = row + dr
                        new_col = col + dc
                        
                        # Check boundaries
                        if 0 <= new_row < self.size and 0 <= new_col < self.size:
                            # Calculate transparency based on distance (farther = more transparent)
                            # Distance 1: 0.6, Distance 1.4: 0.4, Distance 2: 0.3
                            alpha = max(0.3, 1.0 - distance * 0.35)
                            
                            # Update data (use max value to avoid overwriting higher values)
                            if self.data[new_row, new_col] < alpha:
                                self.data[new_row, new_col] = alpha
                                
                                # Calculate grayscale color (semi-transparent effect)
                                gray_value = int(255 * alpha)
                                color = f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"
                                
                                rect_id = self.rectangles[(new_row, new_col)]
                                self.canvas.itemconfig(rect_id, fill=color, outline=color)
            
            if self.on_change_callback:
                self.on_change_callback()
    
    def clear_canvas(self):
        """Clear canvas"""
        self.data = np.zeros((self.size, self.size), dtype=np.float32)
        
        for i in range(self.size):
            for j in range(self.size):
                rect_id = self.rectangles[(i, j)]
                self.canvas.itemconfig(rect_id, fill=CANVAS_BG, outline=CANVAS_BG)
        
        if self.on_change_callback:
            self.on_change_callback()
    
    def get_image_data(self):
        """Get canvas data"""
        return self.data.copy()
    
    def is_empty(self):
        """Check if canvas is empty"""
        return np.sum(self.data) == 0.0
    
    def set_on_change_callback(self, callback):
        """Set callback function when canvas changes"""
        self.on_change_callback = callback


# ==================== Main Application ====================
class numberRecognitionApp:
    """Handwritten number recognition application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten number Recognition - 28×28")
        self.root.configure(bg=DARK_BG)
        
        # Create model
        self.model = numberRecognitionNet()
        self.load_model()
        
        # Training status
        self.is_training = False
        
        # Setup UI
        self.setup_ui()
    
    def load_model(self):
        """Load model"""
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH))
            self.model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        except:
            print(f"Warning: Model file not found, using randomly initialized weights")
    
    def setup_ui(self):
        """Setup user interface"""
        main_frame = tk.Frame(self.root, bg=DARK_BG, padx=20, pady=20)
        main_frame.pack()
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="AI Handwritten number Recognition (28×28)\nNeural Network: 784-256-128-10 (2 Hidden Layers)",
            font=("Arial", 16, "bold"),
            bg=DARK_BG,
            fg=DARK_FG
        )
        title_label.pack(pady=(0, 20))
        
        # Canvas
        self.canvas = DrawingCanvas(main_frame)
        self.canvas.canvas.pack(pady=(0, 20))
        self.canvas.set_on_change_callback(self.on_canvas_change)
        
        # Button container
        button_frame = tk.Frame(main_frame, bg=DARK_BG)
        button_frame.pack(pady=(0, 20))
        
        # Clear button
        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.on_clear_click,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_FG,
            font=("Arial", 12),
            width=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Recognize button
        self.recognize_button = tk.Button(
            button_frame,
            text="Recognize",
            command=self.on_recognize_click,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_FG,
            font=("Arial", 12),
            width=10,
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.recognize_button.pack(side=tk.LEFT, padx=5)
        
        # Train button
        self.train_button = tk.Button(
            button_frame,
            text="Train Model",
            command=self.on_train_click,
            bg="#2d5a2d",
            fg=BUTTON_FG,
            activebackground="#3d6a3d",
            activeforeground=BUTTON_FG,
            font=("Arial", 12),
            width=11,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        # Personal training button
        self.personal_train_button = tk.Button(
            button_frame,
            text="Personal Train",
            command=self.on_personal_train_click,
            bg="#5a2d5a",
            fg=BUTTON_FG,
            activebackground="#6a3d6a",
            activeforeground=BUTTON_FG,
            font=("Arial", 12),
            width=12,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.personal_train_button.pack(side=tk.LEFT, padx=5)
        
        # Result display
        self.result_label = tk.Label(
            main_frame,
            text="Please draw a number on the canvas",
            font=("Arial", 14),
            bg=DARK_BG,
            fg=DARK_FG
        )
        self.result_label.pack()
        
        # Training progress display
        self.progress_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 10),
            bg=DARK_BG,
            fg="#888888"
        )
        self.progress_label.pack(pady=(10, 0))
    
    def on_canvas_change(self):
        """Callback when canvas content changes"""
        if self.canvas.is_empty():
            self.recognize_button.config(state=tk.DISABLED)
            self.result_label.config(text="Please draw a number on the canvas")
        else:
            self.recognize_button.config(state=tk.NORMAL)
    
    def on_clear_click(self):
        """Clear button click"""
        self.canvas.clear_canvas()
        self.result_label.config(text="Please draw a number on the canvas")
    
    def on_recognize_click(self):
        """Recognize button click"""
        try:
            self.result_label.config(text="Recognizing...")
            self.root.update()
            
            canvas_data = self.canvas.get_image_data()
            number, confidence = self.model.predict(canvas_data)
            
            confidence_percent = confidence * 100
            result_text = f"Result: {number}  (Confidence: {confidence_percent:.1f}%)"
            self.result_label.config(text=result_text)
            
        except Exception as e:
            self.result_label.config(text=f"Recognition error: {str(e)}")
            print(f"Error during recognition: {e}")
    
    def on_train_click(self):
        """Train button click"""
        if self.is_training:
            messagebox.showinfo("Notice", "Model is currently training, please wait...")
            return
        
        response = messagebox.askyesno(
            "Confirm Training",
            "Training the model will take a few minutes.\n"
            "First run will download MNIST dataset (~12MB).\n\n"
            "Start training?"
        )
        
        if response:
            # Train in new thread to avoid blocking UI
            thread = threading.Thread(target=self.train_model)
            thread.daemon = True
            thread.start()
    
    def on_personal_train_click(self):
        """Personal training button click"""
        if self.is_training:
            messagebox.showinfo("Notice", "Model is currently training, please wait...")
            return
        
        # Open personal training window
        PersonalTrainingWindow(self.root, self.model, self.on_personal_training_complete)
    
    def train_model(self):
        """Train model"""
        self.is_training = True
        
        # Disable train button
        self.train_button.config(state=tk.DISABLED, text="Training...")
        self.progress_label.config(text="Loading data...")
        
        try:
            # Load MNIST data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            
            train_dataset = datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )
            
            test_dataset = datasets.MNIST(
                root='./data',
                train=False,
                download=True,
                transform=transform
            )
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            # Training parameters
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            epochs = 5
            
            self.progress_label.config(text=f"Starting training, {epochs} epochs...")
            
            # Training loop
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Update progress
                    if (batch_idx + 1) % 200 == 0:
                        progress_text = f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)}"
                        self.progress_label.config(text=progress_text)
                
                # Evaluate
                self.model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in test_loader:
                        output = self.model(data)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                
                accuracy = 100 * correct / total
                avg_loss = running_loss / len(train_loader)
                
                progress_text = f"Epoch {epoch+1}/{epochs} complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                self.progress_label.config(text=progress_text)
                print(progress_text)
            
            # Save model
            torch.save(self.model.state_dict(), MODEL_PATH)
            
            self.progress_label.config(text=f"Training complete! Model saved to {MODEL_PATH}")
            messagebox.showinfo("Training Complete", f"Model training complete!\nFinal accuracy: {accuracy:.2f}%")
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.progress_label.config(text=error_msg)
            messagebox.showerror("Training Error", error_msg)
            print(f"Error during training: {e}")
        
        finally:
            self.is_training = False
            self.train_button.config(state=tk.NORMAL, text="Train Model")
    
    def on_personal_training_complete(self):
        """Personal training complete callback"""
        self.progress_label.config(text="Personal training complete! Model updated")
        messagebox.showinfo("Training Complete", "Personal training complete!\nModel has been optimized for your handwriting style.")


# ==================== Personal Training Window ====================
class PersonalTrainingWindow:
    """Personal training window - let user write each number 3 times"""
    
    def __init__(self, parent, model, on_complete_callback):
        self.parent = parent
        self.model = model
        self.on_complete_callback = on_complete_callback
        
        # Training data storage
        self.training_data = []  # Store (image_data, label) pairs
        self.current_number = 0
        self.current_count = 0
        self.total_per_number = 3
        
        # Create new window
        self.window = tk.Toplevel(parent)
        self.window.title("Personal Training")
        self.window.configure(bg=DARK_BG)
        self.window.grab_set()  # Modal window
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        main_frame = tk.Frame(self.window, bg=DARK_BG, padx=30, pady=30)
        main_frame.pack()
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Personal Training - Teach AI Your Handwriting Style",
            font=("Arial", 16, "bold"),
            bg=DARK_BG,
            fg=DARK_FG
        )
        title_label.pack(pady=(0, 10))
        
        # Description
        desc_label = tk.Label(
            main_frame,
            text="Please write each number (0-9) three times",
            font=("Arial", 12),
            bg=DARK_BG,
            fg="#aaaaaa"
        )
        desc_label.pack(pady=(0, 20))
        
        # Current task prompt
        self.task_label = tk.Label(
            main_frame,
            text=f"Please write number: {self.current_number}  (Round {self.current_count + 1}/{self.total_per_number})",
            font=("Arial", 18, "bold"),
            bg=DARK_BG,
            fg="#4CAF50"
        )
        self.task_label.pack(pady=(0, 20))
        
        # Canvas
        self.canvas = DrawingCanvas(main_frame)
        self.canvas.canvas.pack(pady=(0, 20))
        
        # Button container
        button_frame = tk.Frame(main_frame, bg=DARK_BG)
        button_frame.pack(pady=(0, 10))
        
        # Clear button
        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.on_clear_click,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_FG,
            font=("Arial", 12),
            width=12,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Submit button
        self.submit_button = tk.Button(
            button_frame,
            text="Submit This number",
            command=self.on_submit_click,
            bg="#2d5a2d",
            fg=BUTTON_FG,
            activebackground="#3d6a3d",
            activeforeground=BUTTON_FG,
            font=("Arial", 12),
            width=15,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        # Progress display
        self.progress_label = tk.Label(
            main_frame,
            text=f"Progress: 0/30 (0 numbers completed)",
            font=("Arial", 10),
            bg=DARK_BG,
            fg="#888888"
        )
        self.progress_label.pack(pady=(10, 0))
    
    def on_clear_click(self):
        """Clear button"""
        self.canvas.clear_canvas()
    
    def on_submit_click(self):
        """Submit button"""
        # Check if canvas is empty
        if self.canvas.is_empty():
            messagebox.showwarning("Notice", "Please draw a number first!")
            return
        
        # Save current data
        image_data = self.canvas.get_image_data()
        self.training_data.append((image_data.flatten(), self.current_number))
        
        # Clear canvas
        self.canvas.clear_canvas()
        
        # Update count
        self.current_count += 1
        
        # Check if current number is complete
        if self.current_count >= self.total_per_number:
            self.current_number += 1
            self.current_count = 0
            
            # Check if all numbers are complete
            if self.current_number >= 10:
                self.finish_collection()
                return
        
        # Update UI
        self.update_ui()
    
    def update_ui(self):
        """Update UI display"""
        self.task_label.config(
            text=f"Please write number: {self.current_number}  (Round {self.current_count + 1}/{self.total_per_number})"
        )
        
        total_collected = len(self.training_data)
        numbers_completed = self.current_number
        self.progress_label.config(
            text=f"Progress: {total_collected}/30 ({numbers_completed} numbers completed)"
        )
    
    def finish_collection(self):
        """Finish data collection, start training"""
        self.task_label.config(text="Data collection complete! Training...")
        self.submit_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.progress_label.config(text="Training model with your data...")
        
        # Train in new thread
        thread = threading.Thread(target=self.train_with_personal_data)
        thread.daemon = True
        thread.start()
    
    def train_with_personal_data(self):
        """Train model with personal data"""
        try:
            # Prepare data
            images = []
            labels = []
            for img, label in self.training_data:
                images.append(img)
                labels.append(label)
            
            images_tensor = torch.FloatTensor(images)
            labels_tensor = torch.LongTensor(labels)
            
            # Training parameters
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # Smaller learning rate for fine-tuning
            epochs = 50  # Multiple epochs for thorough learning
            
            self.model.train()
            
            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self.model(images_tensor)
                loss = criterion(output, labels_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    self.progress_label.config(
                        text=f"Training progress: {epoch + 1}/{epochs} - Loss: {loss.item():.4f}"
                    )
            
            # Save model
            torch.save(self.model.state_dict(), MODEL_PATH)
            
            # Complete
            self.window.after(0, self.training_complete)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(error_msg)
            self.window.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.window.after(0, self.window.destroy)
    
    def training_complete(self):
        """Training complete"""
        self.task_label.config(text="✓ Personal training complete!")
        self.progress_label.config(text="Model optimized for your handwriting style")
        
        # Delay window close
        self.window.after(2000, self.close_window)
    
    def close_window(self):
        """Close window"""
        self.on_complete_callback()
        self.window.destroy()


# ==================== Main Program ====================
def main():
    root = tk.Tk()
    app = numberRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

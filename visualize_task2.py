import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Parse the training log
def parse_training_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    kgram_losses = []
    kgram_epochs = []
    lstm_losses = []
    lstm_epochs = []
    
    kgram_samples = []  # (step, epoch, output)
    lstm_samples = []
    
    current_model = None
    
    for i, line in enumerate(lines):
        # Detect which model is training
        if '=== Training model: kgram_mlp_seq ===' in line:
            current_model = 'kgram'
        elif '=== Training model: lstm_seq ===' in line:
            current_model = 'lstm'
        
        # Parse epoch end losses
        if '*** End of Epoch' in line:
            match = re.search(r'Epoch (\d+).*Avg Loss: ([\d.]+)', line)
            if match:
                epoch, loss = int(match.group(1)), float(match.group(2))
                if current_model == 'kgram':
                    kgram_epochs.append(epoch)
                    kgram_losses.append(loss)
                elif current_model == 'lstm':
                    lstm_epochs.append(epoch)
                    lstm_losses.append(loss)
        
        # Parse sample outputs (greedy only for simplicity)
        if 'Generating sample text (greedy)' in line and current_model:
            match = re.search(r'epoch=(\d+), step=(\d+)', line)
            if match:
                epoch, step = int(match.group(1)), int(match.group(2))
                # Next line should have the sample
                if i + 1 < len(lines):
                    sample_line = lines[i + 1]
                    sample_match = re.search(r'Greedy Sample: (.+)', sample_line)
                    if sample_match:
                        sample = sample_match.group(1).strip()
                        if current_model == 'kgram':
                            kgram_samples.append((step, epoch, sample))
                        else:
                            lstm_samples.append((step, epoch, sample))
    
    return {
        'kgram_epochs': kgram_epochs,
        'kgram_losses': kgram_losses,
        'lstm_epochs': lstm_epochs,
        'lstm_losses': lstm_losses,
        'kgram_samples': kgram_samples,
        'lstm_samples': lstm_samples
    }

# Create visualizations
def create_visualizations(data):
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 2, 1)
    if data['kgram_epochs']:
        ax1.plot(data['kgram_epochs'], data['kgram_losses'], 'o-', label='K-gram MLP', linewidth=2, markersize=8, color='#FF6B6B')
    if data['lstm_epochs']:
        ax1.plot(data['lstm_epochs'], data['lstm_losses'], 's-', label='LSTM', linewidth=2, markersize=8, color='#4ECDC4')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Average Loss', fontsize=12)
    ax1.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 4))  # Epochs 1, 2, 3
    
    # 2. Final loss comparison
    ax2 = plt.subplot(2, 2, 2)
    models = []
    final_losses = []
    
    if data['kgram_losses']:
        models.append('K-gram MLP')
        # Use average of last few losses
        final_losses.append(np.mean(data['kgram_losses'][-3:]))
    
    if data['lstm_losses']:
        models.append('LSTM')
        final_losses.append(np.mean(data['lstm_losses'][-3:]))
    
    bars = ax2.bar(models, final_losses, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Final Average Loss', fontsize=12)
    ax2.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. K-gram MLP learning progression
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    ax3.set_title('K-gram MLP Learning Progression', fontsize=14, fontweight='bold', pad=20)
    
    # Select key samples to show progression
    key_samples = []
    if data['kgram_samples']:
        # Get first sample, mid samples, and last sample
        samples = data['kgram_samples']
        indices = [0, len(samples)//4, len(samples)//2, 3*len(samples)//4, -1]
        for idx in indices:
            if 0 <= idx < len(samples) or idx == -1:
                key_samples.append(samples[idx])
    
    y_pos = 0.9
    for step, epoch, sample in key_samples:
        # Truncate long samples
        display_sample = sample[:60] + '...' if len(sample) > 60 else sample
        # Check if it's correct
        is_correct = sample.startswith('0 1 2 3 4 5 6 7 8 9')
        color = 'green' if is_correct else 'red'
        symbol = '✓' if is_correct else '✗'
        
        text = f"Step {step:3d} (Epoch {epoch}): {display_sample} {symbol}"
        ax3.text(0.05, y_pos, text, transform=ax3.transAxes, 
                fontsize=9, family='monospace', color=color, weight='bold' if is_correct else 'normal')
        y_pos -= 0.15
    
    # 4. LSTM learning progression
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    ax4.set_title('LSTM Learning Progression', fontsize=14, fontweight='bold', pad=20)
    
    # Select key samples
    key_samples = []
    if data['lstm_samples']:
        samples = data['lstm_samples']
        indices = [0, len(samples)//4, len(samples)//2, 3*len(samples)//4, -1] if len(samples) > 4 else [0, -1]
        for idx in indices:
            if 0 <= idx < len(samples) or idx == -1:
                key_samples.append(samples[idx])
    
    y_pos = 0.9
    for step, epoch, sample in key_samples:
        display_sample = sample[:60] + '...' if len(sample) > 60 else sample
        is_correct = sample.startswith('0 1 2 3 4 5 6 7 8 9')
        color = 'green' if is_correct else 'red'
        symbol = '✓' if is_correct else '✗'
        
        text = f"Step {step:3d} (Epoch {epoch}): {display_sample} {symbol}"
        ax4.text(0.05, y_pos, text, transform=ax4.transAxes,
                fontsize=9, family='monospace', color=color, weight='bold' if is_correct else 'normal')
        y_pos -= 0.15
    
    plt.tight_layout()
    plt.savefig('TASK2_VISUALIZATION.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as TASK2_VISUALIZATION.png")

if __name__ == '__main__':
    print("Parsing training log...")
    data = parse_training_log('TASK2_TRAINING_LOG.txt')
    print(f"Found {len(data['kgram_losses'])} K-gram epoch losses and {len(data['lstm_losses'])} LSTM epoch losses")
    print(f"Found {len(data['kgram_samples'])} K-gram samples and {len(data['lstm_samples'])} LSTM samples")
    
    print("\nCreating visualizations...")
    create_visualizations(data)


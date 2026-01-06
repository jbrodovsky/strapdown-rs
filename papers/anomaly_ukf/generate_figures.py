#!/usr/bin/env python3
"""
Generate supplementary figures for the geophysical navigation presentation.

This script creates:
1. System architecture block diagram
2. Summary statistics bar chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Temple University color scheme
TEMPLE_CHERRY = '#A41E35'
TEMPLE_GRAY = '#646464'
TEMPLE_LIGHT = '#F0E5E8'

def create_system_architecture():
    """Create a block diagram comparing standard vs geophysically-aided INS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, title, use_geo in [(ax1, 'Standard GNSS-Aided INS', False),
                                (ax2, 'Geophysically-Aided INS', True)]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', color=TEMPLE_CHERRY, pad=20)
        
        # Sensor inputs (left side)
        sensors_y = 7
        box_height = 0.8
        
        # IMU input
        imu_box = FancyBboxPatch((0.5, sensors_y), 1.5, box_height,
                                  boxstyle="round,pad=0.1", 
                                  facecolor=TEMPLE_LIGHT, 
                                  edgecolor=TEMPLE_CHERRY, linewidth=2)
        ax.add_patch(imu_box)
        ax.text(1.25, sensors_y + box_height/2, 'IMU\n(Accel/Gyro)', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # GNSS input
        gnss_box = FancyBboxPatch((0.5, sensors_y - 2), 1.5, box_height,
                                   boxstyle="round,pad=0.1", 
                                   facecolor=TEMPLE_LIGHT, 
                                   edgecolor=TEMPLE_CHERRY, linewidth=2)
        ax.add_patch(gnss_box)
        ax.text(1.25, sensors_y - 2 + box_height/2, 'GNSS\n(Degraded)', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Barometer input
        baro_box = FancyBboxPatch((0.5, sensors_y - 4), 1.5, box_height,
                                   boxstyle="round,pad=0.1", 
                                   facecolor=TEMPLE_LIGHT, 
                                   edgecolor=TEMPLE_CHERRY, linewidth=2)
        ax.add_patch(baro_box)
        ax.text(1.25, sensors_y - 4 + box_height/2, 'Barometer', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Geophysical inputs (only for aided system)
        if use_geo:
            geo_y = sensors_y - 6
            grav_box = FancyBboxPatch((0.5, geo_y), 1.5, box_height,
                                       boxstyle="round,pad=0.1", 
                                       facecolor='#FFD700', 
                                       edgecolor=TEMPLE_CHERRY, linewidth=2)
            ax.add_patch(grav_box)
            ax.text(1.25, geo_y + box_height/2, 'Gravity\nAnomaly', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
            
            mag_box = FancyBboxPatch((0.5, geo_y - 1.5), 1.5, box_height,
                                      boxstyle="round,pad=0.1", 
                                      facecolor='#FFD700', 
                                      edgecolor=TEMPLE_CHERRY, linewidth=2)
            ax.add_patch(mag_box)
            ax.text(1.25, geo_y - 1.5 + box_height/2, 'Magnetic\nAnomaly', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        # UKF Filter (center)
        ukf_x, ukf_y = 4.25, 3.5
        ukf_width, ukf_height = 2.5, 3
        ukf_box = FancyBboxPatch((ukf_x, ukf_y), ukf_width, ukf_height,
                                  boxstyle="round,pad=0.15", 
                                  facecolor=TEMPLE_CHERRY, 
                                  edgecolor='black', linewidth=2.5)
        ax.add_patch(ukf_box)
        
        state_text = '15-State UKF' if not use_geo else '16-State UKF\n(+Geo Bias)'
        ax.text(ukf_x + ukf_width/2, ukf_y + ukf_height/2 + 0.5, state_text, 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(ukf_x + ukf_width/2, ukf_y + ukf_height/2 - 0.5, 
                '9 Nav States\n6 IMU Bias', 
                ha='center', va='center', fontsize=8, color='white')
        
        # Outputs (right side)
        output_x = 8.5
        output_box = FancyBboxPatch((output_x, 4.5), 1.5, 1.5,
                                     boxstyle="round,pad=0.1", 
                                     facecolor=TEMPLE_LIGHT, 
                                     edgecolor=TEMPLE_CHERRY, linewidth=2)
        ax.add_patch(output_box)
        ax.text(output_x + 0.75, 5.25, 'Navigation\nSolution', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(output_x + 0.75, 4.8, '(Pos, Vel, Att)', 
                ha='center', va='center', fontsize=7)
        
        # Arrows
        arrow_props = dict(arrowstyle='->', lw=2, color=TEMPLE_GRAY)
        
        # IMU to UKF
        ax.annotate('', xy=(ukf_x, ukf_y + ukf_height - 0.5), 
                    xytext=(2.2, sensors_y + box_height/2),
                    arrowprops=arrow_props)
        
        # GNSS to UKF
        ax.annotate('', xy=(ukf_x, ukf_y + ukf_height/2), 
                    xytext=(2.2, sensors_y - 2 + box_height/2),
                    arrowprops=arrow_props)
        
        # Baro to UKF
        ax.annotate('', xy=(ukf_x, ukf_y + 0.5), 
                    xytext=(2.2, sensors_y - 4 + box_height/2),
                    arrowprops=arrow_props)
        
        if use_geo:
            # Gravity to UKF
            ax.annotate('', xy=(ukf_x, ukf_y + 0.2), 
                        xytext=(2.2, geo_y + box_height/2),
                        arrowprops=arrow_props)
            
            # Magnetic to UKF
            ax.annotate('', xy=(ukf_x, ukf_y), 
                        xytext=(2.2, geo_y - 1.5 + box_height/2),
                        arrowprops=arrow_props)
        
        # UKF to Output
        ax.annotate('', xy=(output_x, 5.25), 
                    xytext=(ukf_x + ukf_width, ukf_y + ukf_height/2),
                    arrowprops=arrow_props)
    
    plt.tight_layout()
    plt.savefig('figures/system_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Generated system_architecture.png")
    plt.close()


def create_summary_statistics():
    """Create bar chart of summary statistics across all trajectories."""
    
    # Data from Table 1 in the paper (median errors in meters)
    # Format: trajectory_name: (baseline_error, gravity_error, magnetic_error)
    trajectories = {
        '2023-08-04': (1177, 7, 41),
        '2023-08-06': (120, 47, 60),
        '2023-08-09_124742': (73, 42, 42),
        '2023-08-09_163741': (29, 8659, 8635),  # Outlier
        '2025-03-01_150426': (86, 57, 70),
        '2025-03-01_164639': (147, 60, 152),
        '2025-06-18_15-09': (160, 187, 156),
        '2025-06-18_16-52': (86, 84, 101),
        '2025-06-27': (583, 18, 33),
        '2025-07-04': (101, 87, 84),
        '2025-07-18': (86, 101, 89),
        '2025-07-31': (705, 17, 73),
        '2025-09-26': (52, 78, 51),
        '2025-09-27': (66, 69, 62),
    }
    
    # Remove outlier for clearer visualization
    trajectories_no_outlier = {k: v for k, v in trajectories.items() 
                                if k != '2023-08-09_163741'}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All trajectories
    traj_names = list(trajectories.keys())
    baseline = [trajectories[t][0] for t in traj_names]
    gravity = [trajectories[t][1] for t in traj_names]
    magnetic = [trajectories[t][2] for t in traj_names]
    
    x = np.arange(len(traj_names))
    width = 0.25
    
    ax1.bar(x - width, baseline, width, label='GNSS-Degraded Baseline', 
            color=TEMPLE_GRAY, edgecolor='black', linewidth=0.5)
    ax1.bar(x, gravity, width, label='Gravity-Aided', 
            color=TEMPLE_CHERRY, edgecolor='black', linewidth=0.5)
    ax1.bar(x + width, magnetic, width, label='Magnetic-Aided', 
            color='#D4AF37', edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('Median Position Error (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Navigation Performance Across All Trajectories (Including Outlier)', 
                  fontsize=14, fontweight='bold', color=TEMPLE_CHERRY)
    ax1.set_xticks(x)
    ax1.set_xticklabels(traj_names, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 9000])
    
    # Highlight outlier
    outlier_idx = traj_names.index('2023-08-09_163741')
    ax1.axvspan(outlier_idx - 0.5, outlier_idx + 0.5, alpha=0.2, color='red')
    ax1.text(outlier_idx, 8500, 'Outlier', ha='center', fontsize=9, 
             color='red', fontweight='bold')
    
    # Plot 2: Excluding outlier for detail
    traj_names_no_outlier = list(trajectories_no_outlier.keys())
    baseline_no = [trajectories_no_outlier[t][0] for t in traj_names_no_outlier]
    gravity_no = [trajectories_no_outlier[t][1] for t in traj_names_no_outlier]
    magnetic_no = [trajectories_no_outlier[t][2] for t in traj_names_no_outlier]
    
    x2 = np.arange(len(traj_names_no_outlier))
    
    ax2.bar(x2 - width, baseline_no, width, label='GNSS-Degraded Baseline', 
            color=TEMPLE_GRAY, edgecolor='black', linewidth=0.5)
    ax2.bar(x2, gravity_no, width, label='Gravity-Aided', 
            color=TEMPLE_CHERRY, edgecolor='black', linewidth=0.5)
    ax2.bar(x2 + width, magnetic_no, width, label='Magnetic-Aided', 
            color='#D4AF37', edgecolor='black', linewidth=0.5)
    
    ax2.set_ylabel('Median Position Error (m)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
    ax2.set_title('Navigation Performance (Outlier Excluded)', 
                  fontsize=14, fontweight='bold', color=TEMPLE_CHERRY)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(traj_names_no_outlier, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('figures/summary_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Generated summary_statistics.png")
    plt.close()


if __name__ == '__main__':
    print("Generating supplementary figures for presentation...")
    create_system_architecture()
    create_summary_statistics()
    print("\nAll figures generated successfully!")

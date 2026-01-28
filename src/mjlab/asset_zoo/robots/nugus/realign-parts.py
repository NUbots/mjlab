#!/usr/bin/env python3
"""
Interactive XML position adjuster.

This script helps you visually identify which body positions need adjustment
after rotating STL files. It compares the current XML against a reference.

Usage:
    python interactive_position_fixer.py nugus.xml
"""

import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
import numpy as np


def analyze_body_positions(xml_path):
    """
    Analyze all body positions in the XML and identify potential issues.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print("=" * 70)
    print("Body Position Analysis")
    print("=" * 70)
    print()
    
    bodies = root.findall('.//body[@name]')
    
    for body in bodies:
        name = body.get('name')
        pos = body.get('pos', '0 0 0')
        quat = body.get('quat', '1 0 0 0')
        
        pos_array = np.array([float(x) for x in pos.split()])
        
        print(f"Body: {name}")
        print(f"  Position: {pos}")
        print(f"  Quaternion: {quat}")
        
        # Check for suspiciously large offsets
        if np.linalg.norm(pos_array) > 1.0:
            print(f"  ⚠ Warning: Large offset detected ({np.linalg.norm(pos_array):.3f}m)")
        
        print()


def create_position_adjustment_guide(xml_path):
    """
    Create a guide showing which positions likely need adjustment.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    guide = []
    guide.append("# Position Adjustment Guide")
    guide.append("# After rotating STLs 180° around Z-axis:")
    guide.append("# - X coordinates should flip sign: x → -x")
    guide.append("# - Y coordinates should flip sign: y → -y")
    guide.append("# - Z coordinates stay the same: z → z")
    guide.append("")
    
    bodies = root.findall('.//body[@pos]')
    
    guide.append("## Suggested Position Adjustments:")
    guide.append("")
    
    for body in bodies:
        name = body.get('name', 'unnamed')
        pos_str = body.get('pos')
        pos = np.array([float(x) for x in pos_str.split()])
        
        # For 180° Z rotation: flip X and Y
        pos_suggested = np.array([-pos[0], -pos[1], pos[2]])
        
        if not np.allclose(pos, pos_suggested):
            guide.append(f"Body: {name}")
            guide.append(f"  Current:   pos=\"{pos_str}\"")
            guide.append(f"  Suggested: pos=\"{' '.join(f'{x:.8g}' for x in pos_suggested)}\"")
            guide.append("")
    
    return '\n'.join(guide)


def visualize_coordinate_frames(xml_path):
    """
    Launch MuJoCo viewer with coordinate frames visible.
    """
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print("\n✓ XML loaded successfully")
        print("\nLaunching MuJoCo viewer...")
        print("Controls:")
        print("  - Press 'F' to toggle coordinate frame visualization")
        print("  - Check if body frames are aligned correctly")
        print("  - Look for misaligned parts")
        print("\nClose the viewer when done.")
        
        mujoco.viewer.launch(model)
        
    except Exception as e:
        print(f"\n❌ Error loading XML: {e}")
        print("\nThis likely means positions are severely misaligned.")
        print("Use the transform_xml_positions.py script to fix them.")


def check_symmetry(xml_path):
    """
    Check if left/right side positions are symmetric (useful for humanoids).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print("=" * 70)
    print("Symmetry Check (for humanoid robots)")
    print("=" * 70)
    print()
    
    # Find pairs of left/right bodies
    all_bodies = {body.get('name'): body for body in root.findall('.//body[@name]')}
    
    for name, body in all_bodies.items():
        if 'left' in name.lower():
            # Try to find corresponding right body
            right_name = name.lower().replace('left', 'right')
            if right_name in {k.lower(): k for k in all_bodies.keys()}:
                # Get actual right body name (with correct case)
                right_body_name = [k for k in all_bodies.keys() if k.lower() == right_name][0]
                right_body = all_bodies[right_body_name]
                
                left_pos = np.array([float(x) for x in body.get('pos', '0 0 0').split()])
                right_pos = np.array([float(x) for x in right_body.get('pos', '0 0 0').split()])
                
                print(f"Pair: {name} ↔ {right_body_name}")
                print(f"  Left:  {left_pos}")
                print(f"  Right: {right_pos}")
                
                # Check if symmetric (Y should be opposite, X and Z same)
                expected_right = np.array([left_pos[0], -left_pos[1], left_pos[2]])
                
                if np.allclose(right_pos, expected_right, atol=0.001):
                    print("  ✓ Symmetric")
                else:
                    print(f"  ✗ Asymmetric! Expected: {expected_right}")
                
                print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python interactive_position_fixer.py <xml_file>")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    
    print("\n" + "=" * 70)
    print("Interactive XML Position Fixer")
    print("=" * 70)
    print()

    print(xml_path)
    
    # Step 1: Analyze positions
    print("STEP 1: Analyzing body positions...")
    analyze_body_positions(xml_path)
    
    # Step 2: Check symmetry (for humanoids)
    print("\nSTEP 2: Checking left/right symmetry...")
    check_symmetry(xml_path)
    
    # Step 3: Generate adjustment guide
    print("\nSTEP 3: Generating position adjustment guide...")
    guide = create_position_adjustment_guide(xml_path)
    
    guide_file = xml_path.replace('.xml', '_position_guide.txt')
    with open(guide_file, 'w') as f:
        f.write(guide)
    
    print(f"✓ Adjustment guide saved to: {guide_file}")
    print()
    
    # Step 4: Visual check
    print("STEP 4: Visual verification...")
    response = input("Launch MuJoCo viewer to visually check? (y/n): ")
    
    if response.lower() == 'y':
        visualize_coordinate_frames(xml_path)
    
    print("\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print()
    print("Option A (Recommended): Use automatic transformation")
    print("  python transform_xml_positions.py nugus.xml nugus_fixed.xml --axis z --angle 180")
    print()
    print("Option B: Manual adjustment")
    print(f"  1. Open {guide_file}")
    print("  2. Apply suggested position changes to XML")
    print("  3. Reload and verify in MuJoCo viewer")
    print()
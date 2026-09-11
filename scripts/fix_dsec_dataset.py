import os
import glob
import pickle
import shutil

def main():
    base_dir = "/home/ubuntu/mmood3d/dsec_workspace/data/DSEC-3DOD"
    deep_dir = os.path.join(base_dir, "content/drive/MyDrive/dataset/Ev3DOD/DSEC")
    
    print(f"=== Phase 1: Moving sequence folders ===")
    if os.path.exists(deep_dir):
        items = os.listdir(deep_dir)
        for item in items:
            src = os.path.join(deep_dir, item)
            dst = os.path.join(base_dir, item)
            if not os.path.exists(dst):
                print(f"Moving {item} to base directory...")
                shutil.move(src, dst)
        
        content_dir = os.path.join(base_dir, "content")
        if os.path.exists(content_dir):
            shutil.rmtree(content_dir)
            print("Removed nested empty directories (content/).")
    else:
        print("Nested directory already cleaned.")

    print(f"\n=== Phase 2: Fixing hardcoded paths in PKL files ===")
    pkl_files = glob.glob(os.path.join(base_dir, "**/*.pkl"), recursive=True)
    count_fixed = 0
    
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            needs_update = False
            
            if isinstance(data, list):
                for item in data:
                    if 'image' in item and 'event_0_path' in item['image']:
                        original_path = item['image']['event_0_path']
                        if "/home/jaeyoung/" in original_path:
                            parts = original_path.split('/')
                            if 'raw_events' in parts:
                                raw_idx = parts.index('raw_events')
                                relative_path = "/".join(parts[raw_idx-1:])
                                item['image']['event_0_path'] = relative_path
                                needs_update = True
                            
            if needs_update:
                with open(pkl_path, 'wb') as f:
                    pickle.dump(data, f)
                count_fixed += 1
                
        except Exception as e:
            print(f"Error processing {pkl_path}: {e}")
            
    print(f"Fixed {count_fixed} PKL files containing hardcoded paths.")

if __name__ == '__main__':
    main()

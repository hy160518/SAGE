
import re
from pathlib import Path

def fix_unicode_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace('\u2713', '[OK]')
        content = content.replace('\u2717', '[ERROR]')
        content = content.replace('✓', '[OK]')
        content = content.replace('✗', '[ERROR]')
        content = content.replace('⭐', '*')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[OK] FIXED: {file_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] FAILED: {file_path.name}: {e}")
        return False

def main():
    script_dir = Path(__file__).parent
    
    files_to_fix = [
        script_dir / 'src' / 'utils' / 'annotation_generator.py',
        script_dir / 'eval' / 'downstream_task_metrics.py',
        script_dir / 'eval' / 'asr_performance.py',
        script_dir / 'eval' / 'drug_extraction_accuracy.py',
        script_dir / 'eval' / 'run_all_evaluations.py',
    ]
    
    print("=" * 80)
    print("FIXING Unicode Encoding Issues".center(80))
    print("=" * 80)
    
    success_count = 0
    for file_path in files_to_fix:
        if file_path.exists():
            if fix_unicode_in_file(file_path):
                success_count += 1
        else:
            print(f"[SKIP] FILE NOT FOUND: {file_path.name}")
    
    print("=" * 80)
    print(f"FIXING COMPLETED: {success_count}/{len(files_to_fix)} FILES")
    print("=" * 80)

if __name__ == '__main__':
    main()

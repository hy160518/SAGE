import os
import json
import time
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import pickle

from src.data.intake import load_chat_messages, save_quality_report
from src.data.pseudonymizer import Pseudonymizer

DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "annotations", "Case_A_chat_messages.json")
DEFAULT_REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "reports", "data_quality.json")
DEFAULT_CLEAN = os.path.join(os.path.dirname(__file__), "..", "..", "output", "clean", "Case_A_chat_messages.clean.json")
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "checkpoints")


class PipelineStage(Enum):
    INIT = "initialization"
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    PSEUDONYMIZATION = "pseudonymization"
    DISPATCH = "dispatch"
    PROCESSING = "processing"
    FUSION = "fusion"
    QUALITY_CHECK = "quality_check"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class PipelineCheckpoint:
    stage: PipelineStage
    timestamp: str
    data: Dict[str, Any]
    progress: float  # 0.0 - 1.0
    message: str


class PipelineStatistics:
    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.stage_times = {}
        self.errors = []
        self.warnings = []
        self.data_counts = {
            'input': 0,
            'valid': 0,
            'processed': 0,
            'output': 0,
            'failed': 0
        }
        self.checkpoints = []
    
    def to_dict(self) -> Dict:
        duration = 0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': duration,
            'stage_times': {k: v for k, v in self.stage_times.items()},
            'data_counts': dict(self.data_counts),
            'errors': self.errors,
            'warnings': self.warnings,
            'checkpoint_count': len(self.checkpoints)
        }


class PipelineRunner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.stats = PipelineStatistics()
        self.current_stage = PipelineStage.INIT
        self.checkpoint_dir = self.config.get('checkpoint_dir', DEFAULT_CHECKPOINT)
        self.enable_checkpoint = self.config.get('enable_checkpoint', True)
        self.progress_callback: Optional[Callable] = None
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def set_progress_callback(self, callback: Callable[[PipelineStage, float, str], None]):
        self.progress_callback = callback
    
    def _report_progress(self, stage: PipelineStage, progress: float, message: str):
        self.current_stage = stage
        print(f"[{stage.value}] {progress:.1%} - {message}")
        
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
    
    def _save_checkpoint(self, stage: PipelineStage, data: Dict[str, Any], progress: float, message: str):
        if not self.enable_checkpoint:
            return
        
        checkpoint = PipelineCheckpoint(
            stage=stage,
            timestamp=datetime.now().isoformat(),
            data=data,
            progress=progress,
            message=message
        )
        
        self.stats.checkpoints.append({
            'stage': stage.value,
            'timestamp': checkpoint.timestamp,
            'progress': progress,
            'message': message
        })
        
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{stage.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  ✓ Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            print(f"  ⚠ Checkpoint save failed: {str(e)}")
    
    def _load_latest_checkpoint(self) -> Optional[PipelineCheckpoint]:
        if not self.enable_checkpoint or not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pkl')
        ]
        
        if not checkpoint_files:
            return None
        
        checkpoint_files.sort(reverse=True)
        latest_file = os.path.join(self.checkpoint_dir, checkpoint_files[0])
        
        try:
            with open(latest_file, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"✓ Loaded checkpoint from {checkpoint.stage.value} ({checkpoint.timestamp})")
            return checkpoint
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {str(e)}")
            return None
    
    def run(self, 
            input_path: str = DEFAULT_INPUT,
            salt: str = "experiment_salt_v1",
            resume_from_checkpoint: bool = False,
            run_full_pipeline: bool = False) -> Dict[str, Any]:

        self.stats.start_time = datetime.now()
        stage_start = time.time()
        
        self._report_progress(PipelineStage.INIT, 0.0, "Initializing pipeline...")
        
        if resume_from_checkpoint:
            checkpoint = self._load_latest_checkpoint()
            if checkpoint:
                return self._resume_from_checkpoint(checkpoint, input_path, salt, run_full_pipeline)
        
        stage_start = time.time()
        self._report_progress(PipelineStage.DATA_LOADING, 0.1, f"Loading data from {input_path}")
        
        valids, errors = load_chat_messages(input_path)
        self.stats.data_counts['input'] = len(valids) + len(errors)
        self.stats.data_counts['valid'] = len(valids)
        self.stats.stage_times['data_loading'] = time.time() - stage_start
        
        self._save_checkpoint(
            PipelineStage.DATA_LOADING,
            {'valids': valids, 'errors': errors},
            0.2,
            f"Loaded {len(valids)} valid items, {len(errors)} errors"
        )
        
        stage_start = time.time()
        self._report_progress(PipelineStage.DATA_VALIDATION, 0.2, "Validating data quality...")
        
        save_quality_report(DEFAULT_REPORT, len(valids), errors)
        
        if len(valids) == 0:
            self.stats.end_time = datetime.now()
            return {
                "status": "failed",
                "reason": "no_valid_data",
                "valid_count": 0,
                "error_count": len(errors),
                "statistics": self.stats.to_dict()
            }
        
        self.stats.stage_times['data_validation'] = time.time() - stage_start
        
        stage_start = time.time()
        self._report_progress(PipelineStage.PSEUDONYMIZATION, 0.3, "Pseudonymizing sensitive data...")
        
        pseudo = Pseudonymizer(salt, enable_context=True)
        cleaned = []
        
        for idx, item in enumerate(valids):
            new_item = dict(item)
            new_item["content"] = pseudo.transform(item.get("content", ""))
            cleaned.append(new_item)
            
            if (idx + 1) % 100 == 0 or idx == len(valids) - 1:
                progress = 0.3 + 0.2 * (idx + 1) / len(valids)
                self._report_progress(
                    PipelineStage.PSEUDONYMIZATION,
                    progress,
                    f"Processed {idx + 1}/{len(valids)} items"
                )
        
        self.stats.data_counts['processed'] = len(cleaned)
        self.stats.stage_times['pseudonymization'] = time.time() - stage_start
        
        pseudo_stats = pseudo.get_statistics()
        
        self._save_checkpoint(
            PipelineStage.PSEUDONYMIZATION,
            {'cleaned': cleaned, 'pseudo_stats': pseudo_stats},
            0.5,
            f"Pseudonymized {len(cleaned)} items"
        )
        
        stage_start = time.time()
        self._report_progress(PipelineStage.EXPORT, 0.8, "Exporting cleaned data...")
        
        os.makedirs(os.path.dirname(DEFAULT_CLEAN), exist_ok=True)
        with open(DEFAULT_CLEAN, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
        self.stats.data_counts['output'] = len(cleaned)
        self.stats.stage_times['export'] = time.time() - stage_start
        
        dispatch_results = None
        fusion_results = None
        
        if run_full_pipeline:
            dispatch_results = self._run_dispatcher(cleaned)
            if dispatch_results:
                fusion_results = self._run_fusion(dispatch_results)
        
        self.stats.end_time = datetime.now()
        self._report_progress(PipelineStage.COMPLETE, 1.0, "Pipeline completed successfully!")
        
        result = {
            "status": "success",
            "valid_count": len(valids),
            "error_count": len(errors),
            "processed_count": len(cleaned),
            "clean_path": DEFAULT_CLEAN,
            "report_path": DEFAULT_REPORT,
            "pseudonymization_stats": pseudo_stats,
            "statistics": self.stats.to_dict()
        }
        
        if dispatch_results:
            result['dispatch_results'] = dispatch_results
        if fusion_results:
            result['fusion_results'] = fusion_results
        
        return result
    
    def _run_dispatcher(self, data: List[Dict]) -> Optional[Dict]:
        stage_start = time.time()
        self._report_progress(PipelineStage.DISPATCH, 0.55, "Running dispatcher...")
        
        try:

            results = {'status': 'dispatcher_placeholder', 'count': len(data)}
            self.stats.stage_times['dispatch'] = time.time() - stage_start
            
            self._save_checkpoint(
                PipelineStage.DISPATCH,
                {'dispatch_results': results},
                0.65,
                f"Dispatched {len(data)} items"
            )
            
            return results
        except Exception as e:
            self.stats.errors.append(f"Dispatcher failed: {str(e)}")
            return None
    
    def _run_fusion(self, dispatch_results: Dict) -> Optional[Dict]:
        stage_start = time.time()
        self._report_progress(PipelineStage.FUSION, 0.7, "Running UIDN fusion...")
        
        try:

            results = {'status': 'fusion_placeholder'}
            self.stats.stage_times['fusion'] = time.time() - stage_start
            
            self._save_checkpoint(
                PipelineStage.FUSION,
                {'fusion_results': results},
                0.75,
                "UIDN fusion completed"
            )
            
            return results
        except Exception as e:
            self.stats.errors.append(f"Fusion failed: {str(e)}")
            return None
    
    def _resume_from_checkpoint(self, checkpoint: PipelineCheckpoint, 
                                input_path: str, salt: str, 
                                run_full_pipeline: bool) -> Dict[str, Any]:
        print(f"Resuming from stage: {checkpoint.stage.value}")

        if checkpoint.stage == PipelineStage.DATA_LOADING:
            return self.run(input_path, salt, resume_from_checkpoint=False, run_full_pipeline=run_full_pipeline)
        
        return {"status": "resumed", "checkpoint_stage": checkpoint.stage.value}


# 保留旧接口兼容性
def run(input_path: str = DEFAULT_INPUT, salt: str = "experiment_salt_v1") -> Dict[str, Any]:
    runner = PipelineRunner()
    return runner.run(input_path, salt, resume_from_checkpoint=False, run_full_pipeline=False)


if __name__ == "__main__":
    runner = PipelineRunner()
    result = runner.run(run_full_pipeline=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


class TaskAnalyzer:
    @staticmethod
    def analyze_task(task: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {
            'complexity': 'unknown',
            'primary_modality': None,
            'secondary_modalities': [],
            'confidence': 0.0,
            'reason': '',
            'requires_verification': False
        }

        if 'file_path' in task:
            file_path = task['file_path'].lower()
            if any(ext in file_path for ext in ['.mp3', '.wav', '.m4a', '.amr']):
                analysis['primary_modality'] = 'voice'
            elif any(ext in file_path for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                analysis['primary_modality'] = 'image'
        
        if 'content' in task and isinstance(task['content'], str):
            if analysis['primary_modality'] is None:
                analysis['primary_modality'] = 'text'
            else:
                analysis['secondary_modalities'].append('text')
        
        content = str(task.get('content', '')) + str(task.get('metadata', {}))
        
        complexity_score = 0
        if any(keyword in content.lower() for keyword in ['drug', 'medication', 'medicine', '化学', '药物']):
            complexity_score += 2
            analysis['reason'] = 'Contains entity extraction keywords'
        
        if len(content) > 500:
            complexity_score += 1
            analysis['reason'] = 'Long text content'
        
        if 'attachments' in task and task['attachments']:
            complexity_score += 1
            analysis['secondary_modalities'].extend(['image', 'voice'])
        
        if complexity_score >= 2:
            analysis['complexity'] = 'high'
            analysis['confidence'] = 0.8
            analysis['requires_verification'] = True
        elif complexity_score == 1:
            analysis['complexity'] = 'medium'
            analysis['confidence'] = 0.7
        else:
            analysis['complexity'] = 'low'
            analysis['confidence'] = 0.9
        
        return analysis


class WorkerPerformanceTracker:
    """Track worker performance to make intelligent routing decisions"""
    
    def __init__(self):
        self.performance = {
            'text': {'success': 0, 'fail': 0, 'total_time': 0.0},
            'voice': {'success': 0, 'fail': 0, 'total_time': 0.0},
            'image': {'success': 0, 'fail': 0, 'total_time': 0.0}
        }
        self.recent_results = defaultdict(list)  # Track last 10 results per modality
    
    def record_success(self, modality: str, processing_time: float, quality_score: float):
        self.performance[modality]['success'] += 1
        self.performance[modality]['total_time'] += processing_time
        self.recent_results[modality].append({
            'status': 'success',
            'quality': quality_score,
            'time': processing_time
        })
        self._trim_history(modality)
    
    def record_failure(self, modality: str, error_type: str):
        self.performance[modality]['fail'] += 1
        self.recent_results[modality].append({
            'status': 'fail',
            'error': error_type
        })
        self._trim_history(modality)
    
    def _trim_history(self, modality: str, max_size: int = 10):
        if len(self.recent_results[modality]) > max_size:
            self.recent_results[modality] = self.recent_results[modality][-max_size:]
    
    def get_success_rate(self, modality: str) -> float:
        perf = self.performance[modality]
        total = perf['success'] + perf['fail']
        return perf['success'] / total if total > 0 else 0.5
    
    def get_avg_quality(self, modality: str) -> float:
        recent = [r.get('quality', 0.5) for r in self.recent_results[modality] if r.get('status') == 'success']
        return sum(recent) / len(recent) if recent else 0.5
    
    def get_health_score(self, modality: str) -> float:
        success_rate = self.get_success_rate(modality)
        avg_quality = self.get_avg_quality(modality)
        return success_rate * 0.6 + avg_quality * 0.4


class IntelligentWorkerSelector:    
    def __init__(self, performance_tracker: WorkerPerformanceTracker):
        self.performance_tracker = performance_tracker
    
    def select_worker(self, task_analysis: Dict[str, Any], available_workers: List[str]) -> Tuple[str, Dict[str, Any]]:
        primary = task_analysis.get('primary_modality')
        
        decision = {
            'selected_worker': None,
            'alternative_workers': [],
            'decision_strategy': None,
            'reasoning': ''
        }
        
        if primary not in available_workers:
            decision['selected_worker'] = available_workers[0]
            decision['decision_strategy'] = 'rule_based_fallback'
            decision['reasoning'] = f'{primary} worker not available, using {available_workers[0]}'
            return available_workers[0], decision
        
        complexity = task_analysis.get('complexity', 'medium')
        
        if complexity == 'high' and task_analysis.get('requires_verification'):
            if len(available_workers) > 1:
                primary_worker = self._select_best_worker(primary, available_workers)
                alternatives = [w for w in available_workers if w != primary_worker]
                
                decision['selected_worker'] = primary_worker
                decision['alternative_workers'] = alternatives
                decision['decision_strategy'] = 'high_complexity_multi_verification'
                decision['reasoning'] = f'High complexity task ({task_analysis["reason"]}), primary: {primary_worker}, will verify with {alternatives}'
            else:
                decision['selected_worker'] = primary
                decision['decision_strategy'] = 'high_complexity_single'
                decision['reasoning'] = 'High complexity but only one worker available'
        else:
            best_worker = self._select_best_worker(primary, available_workers)
            decision['selected_worker'] = best_worker
            decision['decision_strategy'] = 'health_based_selection'
            health = self.performance_tracker.get_health_score(best_worker)
            decision['reasoning'] = f'Selected {best_worker} (health score: {health:.2f})'
        
        return decision['selected_worker'], decision
    
    def _select_best_worker(self, preferred_worker: str, available_workers: List[str]) -> str:
        if preferred_worker in available_workers:
            health_scores = {w: self.performance_tracker.get_health_score(w) for w in available_workers}
            if health_scores[preferred_worker] > 0.3:
                return preferred_worker
        
        best_worker = max(available_workers, 
                         key=lambda w: self.performance_tracker.get_health_score(w))
        return best_worker
    
    def decide_retry_strategy(self, task: Dict[str, Any], error_type: str, 
                            primary_worker: str, alternatives: List[str]) -> Dict[str, Any]:
        decision = {
            'should_retry': False,
            'strategy': None,
            'next_worker': None,
            'reason': ''
        }
        
        if 'timeout' in error_type.lower():
            if alternatives and len(alternatives) > 0:
                decision['should_retry'] = True
                decision['strategy'] = 'switch_worker'
                decision['next_worker'] = alternatives[0]
                decision['reason'] = 'Timeout occurred, switching to alternative worker'
            else:
                decision['should_retry'] = False
                decision['reason'] = 'Timeout and no alternative worker'
        
        elif 'api_error' in error_type.lower():
            decision['should_retry'] = True
            decision['strategy'] = 'retry_same'
            decision['next_worker'] = primary_worker
            decision['reason'] = 'API error, retrying same worker'
        
        elif 'quality' in error_type.lower():
            if alternatives:
                decision['should_retry'] = True
                decision['strategy'] = 'switch_worker'
                decision['next_worker'] = alternatives[0]
                decision['reason'] = 'Quality issue, switching worker'
            else:
                decision['should_retry'] = False
                decision['reason'] = 'Quality issue but no alternative worker'
        
        else:
            decision['should_retry'] = False
            decision['reason'] = f'Unknown error type: {error_type}'
        
        return decision


class TaskDispatcher:
    def __init__(self, config: Dict[str, Any], processors: Dict[str, Any]):
        self.config = config
        self.processors = processors
        
        self.max_workers = config.get('system', {}).get('max_workers', 4)
        self.enable_parallel = config.get('system', {}).get('enable_parallel', True)
        self.max_retries = config.get('system', {}).get('max_retries', 2)
        
        self.task_analyzer = TaskAnalyzer()
        self.performance_tracker = WorkerPerformanceTracker()
        self.worker_selector = IntelligentWorkerSelector(self.performance_tracker)
        
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'processing_time': {},
            'processor_usage': {'text': 0, 'voice': 0, 'image': 0},
            'retry_count': 0,
            'multi_worker_verifications': 0,
            'decision_log': []
        }
    
    def dispatch_all(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        print("="*60)
        print("Starting Agent-based Task Dispatch...")
        print(f"Text items: {len(data.get('text_entries', []))}")
        print(f"Voice items: {len(data.get('voice_entries', []))}")
        print(f"Image items: {len(data.get('image_entries', []))}")
        print("="*60)
        
        results = {}
        start_time = time.time()
        
        if self.enable_parallel:
            results = self._dispatch_parallel(data)
        else:
            results = self._dispatch_sequential(data)
        
        total_time = time.time() - start_time
        self.stats['total_processing_time'] = total_time
        
        self._print_statistics()
        
        return results
    
    def _dispatch_parallel(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_modality = {}
            
            # Dispatch each modality
            for modality in ['text', 'voice', 'image']:
                entries_key = f'{modality}_entries'
                if data.get(entries_key):
                    future = executor.submit(
                        self._process_modality,
                        modality,
                        data[entries_key]
                    )
                    future_to_modality[future] = modality
            
            for future in as_completed(future_to_modality):
                modality = future_to_modality[future]
                try:
                    result = future.result()
                    results[f'{modality}_results'] = result
                    print(f"✓ {modality} modality processing completed")
                except Exception as e:
                    print(f"✗ {modality} modality processing failed: {str(e)}")
                    results[f'{modality}_results'] = []
                    self.stats['failed_tasks'] += 1
        
        return results
    
    def _dispatch_sequential(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        results = {}
        
        for modality in ['text', 'voice', 'image']:
            entries_key = f'{modality}_entries'
            if data.get(entries_key):
                results[f'{modality}_results'] = self._process_modality(modality, data[entries_key])
        
        return results
    
    def _process_modality(self, modality: str, entries: List[Dict]) -> List[Dict]:
        processor = self.processors.get(modality)
        
        if not processor:
            print(f"Error: Processor for {modality} not found")
            return []
        
        print(f"\n[{modality.upper()}] Starting processing of {len(entries)} items...")
        start_time = time.time()
        results = []
        
        for idx, entry in enumerate(entries):
            # Agent: Analyze task
            task_analysis = self.task_analyzer.analyze_task(entry)
            
            # Agent: Select worker
            available_workers = [modality]
            selected_worker, decision_info = self.worker_selector.select_worker(
                task_analysis, available_workers
            )
            
            # Log decision
            self.stats['decision_log'].append({
                'task_idx': idx,
                'analysis': task_analysis,
                'decision': decision_info
            })
            
            # Process with retry logic
            result = self._process_single_task_with_retry(
                entry, selected_worker, task_analysis, decision_info
            )
            
            if result:
                results.append(result)
                self.stats['completed_tasks'] += 1
            else:
                self.stats['failed_tasks'] += 1
        
        processing_time = time.time() - start_time
        self.stats['processing_time'][modality] = processing_time
        self.stats['processor_usage'][modality] = len(entries)
        
        print(f"[{modality.upper()}] Completed {len(results)}/{len(entries)} items in {processing_time:.2f}s")
        
        return results
    
    def _process_single_task_with_retry(self, task: Dict[str, Any], worker: str,
                                       task_analysis: Dict[str, Any],
                                       decision_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single task with intelligent retry"""
        processor = self.processors.get(worker)
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Process
                result = processor.process_batch([task])
                
                if result and len(result) > 0:
                    # Success
                    result_item = result[0]
                    
                    # Record performance (assume quality score from result or default to 0.8)
                    quality_score = result_item.get('quality_score', 0.8)
                    self.performance_tracker.record_success(
                        worker, 0.1, quality_score
                    )
                    
                    return result_item
                else:
                    # Empty result
                    raise Exception("Processor returned empty result")
            
            except Exception as e:
                error_type = str(type(e).__name__)
                
                # Agent: Decide retry strategy
                alternatives = decision_info.get('alternative_workers', [])
                retry_decision = self.worker_selector.decide_retry_strategy(
                    task, error_type, worker, alternatives
                )
                
                self.performance_tracker.record_failure(worker, error_type)
                
                if retry_decision['should_retry'] and retry_count < self.max_retries:
                    self.stats['retry_count'] += 1
                    retry_count += 1
                    
                    if retry_decision['strategy'] == 'switch_worker':
                        worker = retry_decision['next_worker']
                        processor = self.processors.get(worker)
                        print(f"  Switching to {worker} for retry")
                    else:
                        print(f"  Retrying {worker}...")
                else:
                    print(f"  Failed after {retry_count} attempts: {error_type}")
                    return None
        
        return None
    
    def _print_statistics(self):
        print("\n" + "="*60)
        print("Agent-based Task Processing Statistics")
        print("="*60)
        
        print(f"Total tasks: {self.stats.get('completed_tasks', 0) + self.stats.get('failed_tasks', 0)}")
        print(f"Completed: {self.stats.get('completed_tasks', 0)}")
        print(f"Failed: {self.stats.get('failed_tasks', 0)}")
        print(f"Retries: {self.stats.get('retry_count', 0)}")
        print(f"Total time: {self.stats.get('total_processing_time', 0):.2f}s")
        
        print("\nWorker Health Scores:")
        for modality in ['text', 'voice', 'image']:
            health = self.performance_tracker.get_health_score(modality)
            success_rate = self.performance_tracker.get_success_rate(modality)
            avg_quality = self.performance_tracker.get_avg_quality(modality)
            print(f"  {modality}: health={health:.2f}, success_rate={success_rate:.2%}, avg_quality={avg_quality:.2f}")
        
        print("="*60)
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.stats.copy()
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        return self.stats.get('decision_log', [])
    
    def reset_statistics(self):
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'processing_time': {},
            'processor_usage': {'text': 0, 'voice': 0, 'image': 0},
            'retry_count': 0,
            'multi_worker_verifications': 0,
            'decision_log': []
        }

        }

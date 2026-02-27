import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from src.agents import (
    MasterAgent, ImageAgent, VoiceAgent, TextAgent,
    Task, TaskResult, ModalityType, create_sage_agents
)

class LLMBasedTaskAnalyzer:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {
            'modality': ModalityType.UNKNOWN,
            'confidence': 0.0,
            'probabilities': {
                'text': 0.33,
                'image': 0.33,
                'voice': 0.34
            },
            'reason': '',
            'requires_verification': False
        }

        file_path = str(task.get('file_path', '')).lower()
        content = str(task.get('content', ''))
        metadata = task.get('metadata', {})

        if self.llm_client:
            analysis = await self._llm_classify(task)
        else:
            analysis = self._rule_based_classify(file_path, content, metadata)

        return analysis

    async def _llm_classify(self, task: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Please analyze the following task and infer its modality type (text/image/voice).

        Task info:
        - File path: {task.get('file_path', '')}
        - Content: {str(task.get('content', ''))[:200]}
        - Metadata: {task.get('metadata', {})}

        Output JSON with:
        - modality: inferred modality (text/image/voice)
        - confidence: confidence score (0-1)
        - reason: reasoning
        """

        try:
            result = self.llm_client.generate(prompt)
            return self._parse_llm_result(result)
        except Exception as e:
            return self._rule_based_classify(
                str(task.get('file_path', '')).lower(),
                str(task.get('content', '')),
                task.get('metadata', {})
            )

    def _parse_llm_result(self, result: str) -> Dict[str, Any]:
        try:
            if '{' in result and '}' in result:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                parsed = json.loads(result[json_start:json_end])
                return {
                    'modality': ModalityType(parsed.get('modality', 'text')),
                    'confidence': parsed.get('confidence', 0.5),
                    'probabilities': parsed.get('probabilities', {'text': 0.5, 'image': 0.25, 'voice': 0.25}),
                    'reason': parsed.get('reason', '')
                }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            import logging
            logging.getLogger(__name__).debug(f"LLM result parse error: {e}")
        return self._rule_based_classify('', '', {})

    def _rule_based_classify(self, file_path: str, content: str, metadata: Dict) -> Dict[str, Any]:
        probabilities = {'text': 0.33, 'image': 0.33, 'voice': 0.34}
        modality = ModalityType.UNKNOWN
        confidence = 0.5
        reason = ''

        if any(ext in file_path for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']):
            modality = ModalityType.IMAGE
            probabilities = {'text': 0.1, 'image': 0.8, 'voice': 0.1}
            confidence = 0.9
            reason = 'File extension indicates image'
        elif any(ext in file_path for ext in ['.mp3', '.wav', '.m4a', '.amr', '.aac']):
            modality = ModalityType.VOICE
            probabilities = {'text': 0.1, 'image': 0.1, 'voice': 0.8}
            confidence = 0.9
            reason = 'File extension indicates voice'
        elif metadata.get('message_type') == 'image':
            modality = ModalityType.IMAGE
            probabilities = {'text': 0.2, 'image': 0.7, 'voice': 0.1}
            confidence = 0.8
            reason = 'Message metadata indicates image'
        elif metadata.get('message_type') == 'voice':
            modality = ModalityType.VOICE
            probabilities = {'text': 0.2, 'image': 0.1, 'voice': 0.7}
            confidence = 0.8
            reason = 'Message metadata indicates voice'
        else:
            modality = ModalityType.TEXT
            probabilities = {'text': 0.8, 'image': 0.1, 'voice': 0.1}
            confidence = 0.7
            reason = 'Default to text'

        return {
            'modality': modality,
            'confidence': confidence,
            'probabilities': probabilities,
            'reason': reason,
            'requires_verification': confidence < 0.7
        }

class WorkerPerformanceTracker:
    def __init__(self):
        self.performance = {
            'text': {'success': 0, 'fail': 0, 'total_time': 0.0},
            'voice': {'success': 0, 'fail': 0, 'total_time': 0.0},
            'image': {'success': 0, 'fail': 0, 'total_time': 0.0}
        }
        self.recent_results = defaultdict(list)

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
        primary = task_analysis.get('modality', ModalityType.TEXT).value

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

        if task_analysis.get('requires_verification', False) and len(available_workers) > 1:
            primary_worker = self._select_best_worker(primary, available_workers)
            alternatives = [w for w in available_workers if w != primary_worker]

            decision['selected_worker'] = primary_worker
            decision['alternative_workers'] = alternatives
            decision['decision_strategy'] = 'multi_verification'
            decision['reasoning'] = f"Task requires verification, primary: {primary_worker}, alternatives: {alternatives}"
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

class TaskDispatcher:
    def __init__(self, config: Dict[str, Any], processors: Dict[str, Any], llm_clients: Dict[str, Any] = None):
        self.config = config
        self.processors = processors
        self.llm_clients = llm_clients or {}

        self.max_workers = config.get('system', {}).get('max_workers', 4)
        self.enable_parallel = config.get('system', {}).get('enable_parallel', True)
        self.max_retries = config.get('system', {}).get('max_retries', 2)

        self._init_agents()

        self.task_analyzer = LLMBasedTaskAnalyzer(self.llm_clients.get('chatglm'))
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

    def _init_agents(self):
        self.agents = create_sage_agents(self.config, self.llm_clients)
        self.master_agent = self.agents['master']
        self.image_agent = self.agents['image']
        self.voice_agent = self.agents['voice']
        self.text_agent = self.agents['text']

        async def modality_classifier(task: 'Task') -> Dict[str, Any]:
            task_dict = {
                'file_path': task.content if isinstance(task.content, str) else '',
                'content': task.content,
                'metadata': task.metadata
            }
            return await self.task_analyzer.analyze_task(task_dict)

        self.master_agent.set_modality_classifier(modality_classifier)

    def dispatch_all(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        print("="*60)
        print("Starting LLM Agent-based Task Dispatch...")
        print(f"Text items: {len(data.get('text_entries', []))}")
        print(f"Voice items: {len(data.get('voice_entries', []))}")
        print(f"Image items: {len(data.get('image_entries', []))}")
        print("="*60)

        results = {}
        start_time = time.time()

        tasks = self._convert_to_tasks(data)

        if self.enable_parallel:
            results = self._dispatch_with_agents_parallel(tasks)
        else:
            results = self._dispatch_with_agents_sequential(tasks)

        total_time = time.time() - start_time
        self.stats['total_processing_time'] = total_time

        self._print_statistics()

        return results

    def _convert_to_tasks(self, data: Dict[str, List[Dict]]) -> List[Task]:
        tasks = []
        task_id = 0

        for item in data.get('text_entries', []):
            tasks.append(Task(
                task_id=f"task_{task_id}",
                modality=ModalityType.TEXT,
                content=item.get('content', ''),
                metadata=item
            ))
            task_id += 1

        for item in data.get('voice_entries', []):
            tasks.append(Task(
                task_id=f"task_{task_id}",
                modality=ModalityType.VOICE,
                content=item.get('file_path', ''),
                metadata=item
            ))
            task_id += 1

        for item in data.get('image_entries', []):
            tasks.append(Task(
                task_id=f"task_{task_id}",
                modality=ModalityType.IMAGE,
                content=item.get('file_path', ''),
                metadata=item
            ))
            task_id += 1

        return tasks

    async def _dispatch_with_agents_async(self, tasks: List[Task]) -> List[TaskResult]:
        results = await self.master_agent.dispatch_batch(tasks)
        return results

    def _dispatch_with_agents_parallel(self, tasks: List[Task]) -> Dict[str, List[Dict]]:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._dispatch_with_agents_async(tasks))
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self._dispatch_with_agents_async(tasks))

        organized = {
            'text_results': [],
            'voice_results': [],
            'image_results': [],
            'failed_results': []
        }

        for result in results:
            if result.success:
                metadata = result.metadata or {}
                modality = metadata.get('classified_modality', 'text')

                worker_result = result.result
                if isinstance(worker_result, dict) and 'worker_result' in worker_result:
                    actual_result = worker_result.get('worker_result', {})
                    if isinstance(actual_result, dict):
                        actual_result = actual_result.get('result', worker_result)
                else:
                    actual_result = worker_result

                if modality == 'voice':
                    organized['voice_results'].append(actual_result)
                elif modality == 'image':
                    organized['image_results'].append(actual_result)
                else:
                    organized['text_results'].append(actual_result)
            else:
                organized['failed_results'].append(result.to_dict())

        return organized

    def _dispatch_with_agents_sequential(self, tasks: List[Task]) -> Dict[str, List[Dict]]:
        return self._dispatch_with_agents_parallel(tasks)

    def _print_statistics(self):
        print("\n" + "="*60)
        print("LLM Agent-based Task Processing Statistics")
        print("="*60)

        print(f"Total tasks: {self.stats.get('completed_tasks', 0) + self.stats.get('failed_tasks', 0)}")
        print(f"Completed: {self.stats.get('completed_tasks', 0)}")
        print(f"Failed: {self.stats.get('failed_tasks', 0)}")
        print(f"Retries: {self.stats.get('retry_count', 0)}")
        print(f"Total time: {self.stats.get('total_processing_time', 0):.2f}s")

        print("\nAgent Statistics:")
        for agent_name, agent in [('image', self.image_agent), ('voice', self.voice_agent), ('text', self.text_agent)]:
            stats = agent.get_stats()
            print(f"  {agent_name}: processed={stats['processed']}, success={stats['success']}, failed={stats['failed']}")

        print("="*60)

    def get_statistics(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['agent_stats'] = {
            'image': self.image_agent.get_stats(),
            'voice': self.voice_agent.get_stats(),
            'text': self.text_agent.get_stats()
        }
        return stats

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

import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

class QualityMonitor:
    QUALITY_THRESHOLDS = {
        'excellent': 0.9,
        'good': 0.7,
        'fair': 0.5,
        'poor': 0.3
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        custom_thresholds = config.get('quality', {}).get('thresholds', {})
        if custom_thresholds:
            self.QUALITY_THRESHOLDS.update(custom_thresholds)

        self.stats = {
            'total_checked': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'by_quality_level': defaultdict(int),
            'by_modality': defaultdict(lambda: {'checked': 0, 'passed': 0, 'failed': 0})
        }
        
        self.anomalies = []
        
    def check_result(self, result: Dict[str, Any], modality: str) -> Dict[str, Any]:
        self.stats['total_checked'] += 1
        self.stats['by_modality'][modality]['checked'] += 1
        
        basic_validation = self._validate_basic_format(result)
        
        confidence = result.get('confidence', 0.0)
        confidence_check = self._check_confidence(confidence)
        
        completeness = self._check_completeness(result, modality)

        consistency = self._check_consistency(result, modality)

        quality_score = self._calculate_quality_score(
            basic_validation,
            confidence_check,
            completeness,
            consistency
        )
        
        quality_level = self._get_quality_level(quality_score)

        quality_report = {
            'passed': quality_score >= self.QUALITY_THRESHOLDS['fair'],
            'quality_score': quality_score,
            'quality_level': quality_level,
            'confidence': confidence,
            'checks': {
                'basic_validation': basic_validation,
                'confidence_check': confidence_check,
                'completeness': completeness,
                'consistency': consistency
            },
            'issues': [],
            'warnings': []
        }
        
        self._collect_issues_and_warnings(quality_report, result, modality)
        
        self._update_statistics(quality_report, modality)
        
        if not quality_report['passed']:
            self._record_anomaly(result, quality_report, modality)
        
        return quality_report
    
    def check_batch(self, results: List[Dict[str, Any]], modality: str) -> Dict[str, Any]:
        individual_reports = []
        
        for result in results:
            report = self.check_result(result, modality)
            individual_reports.append(report)
        
        batch_report = self._generate_batch_report(individual_reports, modality)
        
        return batch_report
    
    def _validate_basic_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ['uuid', 'confidence']
        missing_fields = [field for field in required_fields if field not in result]
        
        has_error = 'error' in result
        
        return {
            'valid': len(missing_fields) == 0 and not has_error,
            'missing_fields': missing_fields,
            'has_error': has_error
        }
    
    def _check_confidence(self, confidence: float) -> Dict[str, Any]:
        if not isinstance(confidence, (int, float)):
            return {
                'valid': False,
                'reason': 'Confidence is not of numerical type.'
            }
        
        if not (0.0 <= confidence <= 1.0):
            return {
                'valid': False,
                'reason': f'Confidence out of range [0,1]: {confidence}'
            }
        
        return {
            'valid': True,
            'value': confidence,
            'level': 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.5 else 'low'
        }
    
    def _check_completeness(self, result: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality == 'text':
            has_drugs = 'drugs' in result and result['drugs']
            has_entities = 'entities' in result and result['entities']
            
            complete = has_drugs or has_entities
            
            return {
                'complete': complete,
                'has_drugs': has_drugs,
                'has_entities': has_entities
            }
        
        elif modality == 'voice':
            has_transcription = 'transcription' in result
            has_quality = 'quality' in result
            
            return {
                'complete': has_transcription and has_quality,
                'has_transcription': has_transcription,
                'has_quality': has_quality
            }
        
        elif modality == 'image':
            has_classification = 'classification' in result
            has_details = 'details' in result
            
            return {
                'complete': has_classification and has_details,
                'has_classification': has_classification,
                'has_details': has_details
            }
        
        return {'complete': False}
    
    def _check_consistency(self, result: Dict[str, Any], modality: str) -> Dict[str, Any]:

        issues = []
        
        if modality == 'text':
            drugs = result.get('drugs', {})
            entities = result.get('entities', {})
            
            if drugs and entities:
                entity_drugs = entities.get('drugs', [])
                if drugs and not entity_drugs:
                    issues.append('Extracted drugs but no drug information in entities')
        
        elif modality == 'voice':
            quality = result.get('quality', {})
            transcription = result.get('transcription', {})
            
            quality_score = quality.get('quality_score', 0)
            text = transcription.get('text', '')
            
            if quality_score > 0.8 and len(text) < 10:
                issues.append('High quality score but transcription text is too short')
        
        elif modality == 'image':
            classification = result.get('classification', {})
            ocr_text = result.get('ocr_text', '')
            
            category = classification.get('category', '')
            if category in ['transaction', 'chat_screenshot'] and not ocr_text:
                issues.append('Image category requires text but OCR is empty')
        
        return {
            'consistent': len(issues) == 0,
            'issues': issues
        }
    
    def _calculate_quality_score(self, basic_validation: Dict, confidence_check: Dict,
                                completeness: Dict, consistency: Dict) -> float:

        score = 0.0

        if basic_validation.get('valid', False):
            score += 0.3

        if confidence_check.get('valid', False):
            confidence_value = confidence_check.get('value', 0)
            score += 0.3 * confidence_value

        if completeness.get('complete', False):
            score += 0.2

        if consistency.get('consistent', False):
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_quality_level(self, score: float) -> str:

        if score >= self.QUALITY_THRESHOLDS['excellent']:
            return 'excellent'
        elif score >= self.QUALITY_THRESHOLDS['good']:
            return 'good'
        elif score >= self.QUALITY_THRESHOLDS['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _collect_issues_and_warnings(self, quality_report: Dict, result: Dict, modality: str):
        checks = quality_report['checks']
        
        if not checks['basic_validation']['valid']:
            if checks['basic_validation']['missing_fields']:
                quality_report['issues'].append(
                    f"Missing required fields: {', '.join(checks['basic_validation']['missing_fields'])}"
                )
            if checks['basic_validation']['has_error']:
                quality_report['issues'].append(f"Processing error: {result.get('error', '')}")
        
        if checks['confidence_check'].get('level') == 'low':
            quality_report['warnings'].append('Low confidence, manual review recommended')
        
        if not checks['completeness']['complete']:
            quality_report['warnings'].append('Incomplete data')
        
        if not checks['consistency']['consistent']:
            for issue in checks['consistency']['issues']:
                quality_report['warnings'].append(f"Consistency issue: {issue}")
    
    def _update_statistics(self, quality_report: Dict, modality: str):
        if quality_report['passed']:
            self.stats['passed'] += 1
            self.stats['by_modality'][modality]['passed'] += 1
        else:
            self.stats['failed'] += 1
            self.stats['by_modality'][modality]['failed'] += 1
        
        if quality_report['warnings']:
            self.stats['warnings'] += len(quality_report['warnings'])
        
        level = quality_report['quality_level']
        self.stats['by_quality_level'][level] += 1
    
    def _record_anomaly(self, result: Dict, quality_report: Dict, modality: str):
        anomaly = {
            'uuid': result.get('uuid'),
            'modality': modality,
            'quality_score': quality_report['quality_score'],
            'issues': quality_report['issues'],
            'warnings': quality_report['warnings'],
            'result_summary': self._summarize_result(result)
        }
        
        self.anomalies.append(anomaly)
        
        if len(self.anomalies) > 100:
            self.anomalies = self.anomalies[-100:]
    
    def _summarize_result(self, result: Dict) -> Dict:
        return {
            'uuid': result.get('uuid'),
            'confidence': result.get('confidence'),
            'error': result.get('error', None)
        }
    
    def _generate_batch_report(self, individual_reports: List[Dict], modality: str) -> Dict[str, Any]:
        total = len(individual_reports)
        passed = sum(1 for r in individual_reports if r['passed'])
        failed = total - passed
        
        avg_score = sum(r['quality_score'] for r in individual_reports) / total if total > 0 else 0
        avg_confidence = sum(r['confidence'] for r in individual_reports) / total if total > 0 else 0
        
        quality_distribution = defaultdict(int)
        for r in individual_reports:
            quality_distribution[r['quality_level']] += 1
        
        return {
            'modality': modality,
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'average_quality_score': avg_score,
            'average_confidence': avg_confidence,
            'quality_distribution': dict(quality_distribution),
            'individual_reports': individual_reports
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_checked': self.stats['total_checked'],
            'passed': self.stats['passed'],
            'failed': self.stats['failed'],
            'pass_rate': self.stats['passed'] / self.stats['total_checked'] if self.stats['total_checked'] > 0 else 0,
            'warnings': self.stats['warnings'],
            'by_quality_level': dict(self.stats['by_quality_level']),
            'by_modality': {k: dict(v) for k, v in self.stats['by_modality'].items()},
            'anomalies_count': len(self.anomalies)
        }
    
    def get_anomalies(self) -> List[Dict]:
        return self.anomalies.copy()
    
    def generate_report(self) -> str:
        stats = self.get_statistics()
        
        report = "="*60 + "\n"
        report += "Quality Monitoring Report\n"
        report += "="*60 + "\n\n"
        
        report += f"total_checked: {stats['total_checked']}\n"
        report += f"passed: {stats['passed']}\n"
        report += f"failed: {stats['failed']}\n"
        report += f"pass_rate: {stats['pass_rate']:.2%}\n"
        report += f"warnings: {stats['warnings']}\n\n"
        
        report += "Quality Level Distribution:\n"
        for level, count in stats['by_quality_level'].items():
            report += f"  {level}: {count}\n"
        
        report += "\nStatistics by Modality:\n"
        for modality, modality_stats in stats['by_modality'].items():
            if modality_stats['checked'] > 0:
                pass_rate = modality_stats['passed'] / modality_stats['checked']
                report += f"  {modality}: Checked{modality_stats['checked']}  Passed{modality_stats['passed']}  Pass Rate{pass_rate:.2%}\n"
        
        report += "\n"
        report += "="*60 + "\n"
        
        return report

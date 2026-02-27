import json
import random
from pathlib import Path
from typing import Dict, List, Any

class AnnotationGenerator:
    RANDOM_SEED = 42
    USERS_PER_CASE = 300
    RELATIONSHIPS_PER_CASE = 5000
    MESSAGES_PER_CASE = 30000
    ASR_SAMPLES = 100
    IMAGE_SAMPLES = 433

    ROLES = ['prescriber', 'recipient', 'benign_user']
    ROLE_DISTRIBUTION = {'prescriber': 0.15, 'recipient': 0.25, 'benign_user': 0.60}

    def __init__(self, output_dir: str = 'data/annotations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.RANDOM_SEED)

    def generate(self) -> None:
        for case in ['Case_A', 'Case_B']:
            self._generate_case(case)

    def _generate_case(self, case_name: str) -> None:
        user_roles = self._generate_user_roles()
        self._save_json(case_name, 'user_roles', user_roles)

        relationships = self._generate_relationships()
        self._save_json(case_name, 'user_drug_relations', relationships)

        messages = self._generate_messages(user_roles)
        self._save_json(case_name, 'chat_messages', messages)

        asr_samples = self._generate_asr_samples()
        self._save_json(case_name, 'asr_ground_truth', asr_samples)

        images = self._generate_image_annotations()
        self._save_json(case_name, 'image_annotations', images)

    def _generate_user_roles(self) -> Dict[str, str]:
        roles = {}
        for user_id in range(self.USERS_PER_CASE):
            role = random.choices(
                self.ROLES,
                weights=[
                    self.ROLE_DISTRIBUTION['prescriber'],
                    self.ROLE_DISTRIBUTION['recipient'],
                    self.ROLE_DISTRIBUTION['benign_user']
                ]
            )[0]
            roles[f'user_{user_id}'] = role
        return roles

    def _generate_relationships(self) -> List[Dict[str, Any]]:
        relationships = []
        for rel_id in range(self.RELATIONSHIPS_PER_CASE):
            user1 = f'user_{random.randint(0, self.USERS_PER_CASE - 1)}'
            user2 = f'user_{random.randint(0, self.USERS_PER_CASE - 1)}'

            if user1 == user2:
                continue

            is_related = random.random() < 0.30
            relationships.append({
                'id': rel_id,
                'user1': user1,
                'user2': user2,
                'is_related': is_related,
                'interaction_count': random.randint(1, 50) if is_related else random.randint(0, 5)
            })

        return relationships[:self.RELATIONSHIPS_PER_CASE]

    def _generate_messages(self, user_roles: Dict[str, str]) -> List[Dict[str, Any]]:
        messages = []
        drug_keywords = ['crystal', 'powder', 'pill', 'drug', 'substance', 'package']

        for msg_id in range(self.MESSAGES_PER_CASE):
            user = f'user_{random.randint(0, self.USERS_PER_CASE - 1)}'
            role = user_roles.get(user, 'benign_user')

            if role == 'prescriber':
                drug_mention_prob = 0.40
            elif role == 'recipient':
                drug_mention_prob = 0.30
            else:
                drug_mention_prob = 0.02

            has_drug_mention = random.random() < drug_mention_prob

            text = f'Message {msg_id}'
            if has_drug_mention:
                text += f'. Contains reference to {random.choice(drug_keywords)}'

            messages.append({
                'id': msg_id,
                'user': user,
                'text': text,
                'has_drug_mention': has_drug_mention,
                'timestamp': msg_id
            })

        return messages

    def _generate_asr_samples(self) -> List[Dict[str, Any]]:
        samples = []

        for sample_id in range(self.ASR_SAMPLES):
            reference = f'This is sample {sample_id} for ASR evaluation'

            hypothesis = self._introduce_errors(reference)

            cer = self._calculate_cer(reference, hypothesis)

            samples.append({
                'id': sample_id,
                'reference': reference,
                'hypothesis': hypothesis,
                'cer': round(cer, 4)
            })

        return samples

    def _introduce_errors(self, text: str, error_rate: float = 0.15) -> str:
        chars = list(text)
        for _ in range(int(len(chars) * error_rate)):
            idx = random.randint(0, len(chars) - 1)
            if random.random() < 0.3:
                chars[idx] = ''
            elif random.random() < 0.6:
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            else:
                chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))

        return ''.join(chars)

    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        if not reference:
            return 1.0 if hypothesis else 0.0

        distance = self._levenshtein_distance(reference, hypothesis)
        return distance / len(reference)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return AnnotationGenerator._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _generate_image_annotations(self) -> List[Dict[str, Any]]:
        image_types = ['drug_box', 'drug_list', 'other']
        type_distribution = {'drug_box': 0.60, 'drug_list': 0.20, 'other': 0.20}

        images = []
        for img_id in range(self.IMAGE_SAMPLES):
            img_type = random.choices(
                image_types,
                weights=[
                    type_distribution['drug_box'],
                    type_distribution['drug_list'],
                    type_distribution['other']
                ]
            )[0]

            images.append({
                'id': img_id,
                'filename': f'image_{img_id}.jpg',
                'type': img_type,
                'contains_drugs': img_type != 'other'
            })

        return images

    def _save_json(self, case_name: str, data_type: str, data: Any) -> None:
        filename = self.output_dir / f'{case_name}_{data_type}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def main():
    generator = AnnotationGenerator()
    generator.generate()

if __name__ == '__main__':
    main()

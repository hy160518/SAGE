import time
from typing import Dict, List, Any

from ..processors.text_handler import TextHandler
from ..processors.voice_handler import VoiceHandler
from ..processors.image_handler import ImageHandler


class LlamaIndexWorkflowBaseline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.text_handler = TextHandler(self.config)
        self.voice_handler = VoiceHandler(self.config)
        self.image_handler = ImageHandler(self.config)

        self._llama_available = False
        try:
            from llama_index.core.workflow import Workflow, StartEvent, Step  # type: ignore
            from llama_index.core import VectorStoreIndex, Document  # type: ignore
            self._llama_available = True
            self._Workflow = Workflow
            self._StartEvent = StartEvent
            self._Step = Step
            self._VectorStoreIndex = VectorStoreIndex
            self._Document = Document
        except Exception:
            self._llama_available = False

    def run(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        if self._llama_available:
            return self._run_with_llamaindex(data)
        else:
            return self._run_fallback(data)

    def _run_with_llamaindex(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        Workflow = self._Workflow
        StartEvent = self._StartEvent
        Step = self._Step
        VectorStoreIndex = self._VectorStoreIndex
        Document = self._Document

        context: Dict[str, Any] = {"text": [], "voice": [], "image": []}

        class OCRStep(Step):
            def run(self, ev: Any) -> Any:
                entries = ev.get("image_entries", [])
                results = []
                for entry in entries:
                    res = self.outer.image_handler.extract_text(entry.get('file_path', ''))
                    res['uuid'] = entry.get('uuid')
                    results.append(res)
                return {"ocr_results": results}

        class ASRStep(Step):
            def run(self, ev: Any) -> Any:
                entries = ev.get("voice_entries", [])
                return {"asr_results": self.outer.voice_handler.process_batch(entries)}

        class EntityStep(Step):
            def run(self, ev: Any) -> Any:
                entries = ev.get("text_entries", [])
                return {"entity_results": self.outer.text_handler.process_batch(entries)}

        OCRStep.outer = self
        ASRStep.outer = self
        EntityStep.outer = self

        wf = Workflow()
        wf.add_step("ocr", OCRStep())
        wf.add_step("asr", ASRStep())
        wf.add_step("ent", EntityStep())

        # Start event carries entries
        start_ev = StartEvent(payload={
            "text_entries": data.get('text_entries', []),
            "voice_entries": data.get('voice_entries', []),
            "image_entries": data.get('image_entries', []),
        })

        outputs = wf.run(start_ev)

        docs: List[Document] = []
        for r in outputs.get("ocr_results", []):
            docs.append(Document(text=r.get('text', ''), metadata={"type": "ocr", "uuid": r.get('uuid')}))
        for r in outputs.get("asr_results", []):
            txt = r.get('transcription', {}).get('text', '') if isinstance(r.get('transcription'), dict) else ''
            docs.append(Document(text=txt, metadata={"type": "asr", "uuid": r.get('uuid')}))
        for r in outputs.get("entity_results", []):
            docs.append(Document(text=str(r.get('entities', {})), metadata={"type": "entity", "uuid": r.get('uuid')}))

        if docs:
            _ = VectorStoreIndex.from_documents(docs)

        return {
            'text_results': outputs.get('entity_results', []),
            'voice_results': outputs.get('asr_results', []),
            'image_results': outputs.get('ocr_results', []),
        }

    def _run_fallback(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        start = time.time()
        ocr_results: List[Dict[str, Any]] = []
        for entry in data.get('image_entries', []):
            res = self.image_handler.extract_text(entry.get('file_path', ''))
            res['uuid'] = entry.get('uuid')
            ocr_results.append(res)

        asr_results = self.voice_handler.process_batch(data.get('voice_entries', []))
        ent_results = self.text_handler.process_batch(data.get('text_entries', []))

        _ = time.time() - start
        return {
            'text_results': ent_results,
            'voice_results': asr_results,
            'image_results': ocr_results,
        }

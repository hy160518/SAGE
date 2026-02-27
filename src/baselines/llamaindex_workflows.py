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
            from llama_index.core.workflow import (  # type: ignore
                Workflow, 
                StartEvent, 
                StopEvent, 
                step,
                Event,
            )
            from llama_index.core import VectorStoreIndex, Document, Settings  # type: ignore
            
            self._llama_available = True
            self._Workflow = Workflow
            self._StartEvent = StartEvent
            self._StopEvent = StopEvent
            self._step_decorator = step
            self._Event = Event
            self._VectorStoreIndex = VectorStoreIndex
            self._Document = Document
            self._Settings = Settings
        except Exception as e:
            self._llama_available = False
            self._import_error = str(e)

    def run(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        if not self._llama_available:
            raise ImportError(
                "LlamaIndex-Workflows baseline requires 'llama-index-core' package.\n"
                f"Import failed with: {getattr(self, '_import_error', 'unknown')}\n"
                "Install it with: pip install -r requirements-baselines.txt\n"
                "Or: pip install 'llama-index-core>=0.11.0'"
            )
        return self._run_with_llamaindex(data)

    def _run_with_llamaindex(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        import asyncio
        
        Workflow = self._Workflow
        StartEvent = self._StartEvent
        StopEvent = self._StopEvent
        step = self._step_decorator
        Event = self._Event
        Document = self._Document
        VectorStoreIndex = self._VectorStoreIndex
        Settings = self._Settings
        
        class OCRCompletedEvent(Event):
            ocr_results: List[Dict[str, Any]]
            ocr_docs: List[Any]
        
        class ASRCompletedEvent(Event):
            asr_results: List[Dict[str, Any]]
            asr_docs: List[Any]
        
        class EntityCompletedEvent(Event):
            entity_results: List[Dict[str, Any]]
            entity_docs: List[Any]
        
        class IndexBuiltEvent(Event):
            index: Any
            all_results: Dict[str, List[Dict[str, Any]]]
        
        class ForensicMultiModalWorkflow(Workflow):
            
            def __init__(self, text_handler, voice_handler, image_handler, **kwargs):
                super().__init__(**kwargs)
                self.text_handler = text_handler
                self.voice_handler = voice_handler
                self.image_handler = image_handler
                self.ocr_results = []
                self.asr_results = []
                self.entity_results = []
            
            @step
            async def ocr_step(self, ev):  # type: ignore
                image_entries = ev.get("image_entries", [])
                ocr_results: List[Dict[str, Any]] = []
                ocr_docs: List[Any] = []
                
                for entry in image_entries:
                    res = self.image_handler.extract_text(entry.get('file_path', ''))
                    res['uuid'] = entry.get('uuid')
                    ocr_results.append(res)
                    if res.get('text'):
                        ocr_docs.append(Document(
                            text=res['text'],
                            metadata={"type": "ocr", "uuid": res.get('uuid'), "source": "image"}
                        ))
                
                self.ocr_results = ocr_results
                print(f"[Workflow OCR Step] Processed {len(ocr_results)} images")
                return OCRCompletedEvent(ocr_results=ocr_results, ocr_docs=ocr_docs)
            
            @step
            async def asr_step(self, ev):  # type: ignore
                voice_entries = ev.get("voice_entries", [])
                asr_results = self.voice_handler.process_batch(voice_entries)
                asr_docs: List[Any] = []
                
                for r in asr_results:
                    txt = r.get('transcription', {}).get('text', '') if isinstance(r.get('transcription'), dict) else ''
                    if txt:
                        asr_docs.append(Document(
                            text=txt,
                            metadata={"type": "asr", "uuid": r.get('uuid'), "source": "voice"}
                        ))
                
                self.asr_results = asr_results
                print(f"[Workflow ASR Step] Processed {len(asr_results)} voice entries")
                return ASRCompletedEvent(asr_results=asr_results, asr_docs=asr_docs)
            
            @step
            async def entity_step(self, ev):  # type: ignore
                text_entries = ev.get("text_entries", [])
                entity_results = self.text_handler.process_batch(text_entries)
                entity_docs: List[Any] = []
                
                for r in entity_results:
                    entities_str = str(r.get('entities', {}))
                    if entities_str and entities_str != '{}':
                        entity_docs.append(Document(
                            text=f"Entities: {entities_str}",
                            metadata={"type": "entity", "uuid": r.get('uuid'), "source": "text"}
                        ))
                
                self.entity_results = entity_results
                print(f"[Workflow Entity Step] Processed {len(entity_results)} text entries")
                return EntityCompletedEvent(entity_results=entity_results, entity_docs=entity_docs)
            
            @step
            async def build_index_step(self, ocr_ev, asr_ev, entity_ev):  # type: ignore
                all_docs = ocr_ev.ocr_docs + asr_ev.asr_docs + entity_ev.entity_docs
                
                index = None
                if all_docs:
                    try:
                        index = VectorStoreIndex.from_documents(all_docs)
                        print(f"[Workflow Index Step] Built index with {len(all_docs)} documents")
                    except Exception as e:
                        print(f"[Workflow Index Step] Index creation failed: {e}")
                
                all_results = {
                    'text_results': entity_ev.entity_results,
                    'voice_results': asr_ev.asr_results,
                    'image_results': ocr_ev.ocr_results,
                }
                
                return IndexBuiltEvent(index=index, all_results=all_results)
            
            @step
            async def query_step(self, ev):  # type: ignore
                if ev.index:
                    try:
                        query_engine = ev.index.as_query_engine(similarity_top_k=5)
                        response = await query_engine.aquery(
                            "What drugs and entities were mentioned across all modalities?"
                        )
                        print(f"[Workflow Query Step] Cross-modal query completed")
                        print(f"  Response preview: {str(response)[:200]}...")
                    except Exception as e:
                        print(f"[Workflow Query Step] Query failed: {e}")
                
                return StopEvent(result=ev.all_results)
        
        try:
            from llama_index.llms.openai import OpenAI  # type: ignore
            Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key="sk-placeholder")
        except Exception:
            try:
                from llama_index.core.llms import MockLLM  # type: ignore
                Settings.llm = MockLLM(max_tokens=256)
            except Exception:
                pass
        
        workflow = ForensicMultiModalWorkflow(
            text_handler=self.text_handler,
            voice_handler=self.voice_handler,
            image_handler=self.image_handler,
            timeout=120,
            verbose=False
        )
        
        async def _run_workflow():
            return await workflow.run(
                text_entries=data.get('text_entries', []),
                voice_entries=data.get('voice_entries', []),
                image_entries=data.get('image_entries', [])
            )
        
        try:
            result = asyncio.run(_run_workflow())
            return result
        except RuntimeError as e:
            if 'asyncio.run() cannot be called' in str(e):
                try:
                    import nest_asyncio  # type: ignore
                    nest_asyncio.apply()
                    result = asyncio.run(_run_workflow())
                    return result
                except ImportError:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(_run_workflow())
                    return result
            raise


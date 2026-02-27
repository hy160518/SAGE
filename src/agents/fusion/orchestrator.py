import asyncio
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FusionOrchestrator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.uidn = None
        self.fusion_agents = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.entity_registry = None
        self.relationship_graph = None

    def set_uidn_components(self, entity_registry, relationship_graph):
        self.entity_registry = entity_registry
        self.relationship_graph = relationship_graph

    def register_fusion_agent(self, agent):
        self.fusion_agents.append(agent)

    async def start(self):
        self.running = True
        logger.info("[FusionOrchestrator] Starting event-driven fusion...")

        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[FusionOrchestrator] Error processing event: {e}")

    def stop(self):
        self.running = False

    def emit_event(self, event_type: str, data: Dict[str, Any]):
        asyncio.create_task(self.event_queue.put({
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }))

    async def _process_event(self, event: Dict[str, Any]):
        event_type = event.get('type')
        data = event.get('data', {})

        logger.info(f"[FusionOrchestrator] Processing event: {event_type}")

        if event_type == 'modality_output':
            await self._handle_modality_output(data)
        elif event_type == 'external_data':
            await self._handle_external_data(data)
        elif event_type == 'entity_update':
            await self._handle_entity_update(data)

    async def _handle_modality_output(self, data: Dict[str, Any]):
        modality = data.get('modality')
        result = data.get('result', {})

        if self.entity_registry:
            entities = result.get('entities', [])
            for entity in entities:
                entity['source'] = f"{modality}:{result.get('id', 'unknown')}"
                entity['modality'] = modality
                self.entity_registry.register_entity(entity)

    async def _handle_external_data(self, data: Dict[str, Any]):
        external_records = data.get('records', [])

        if self.entity_registry:
            for record in external_records:
                entity_attrs = {
                    'name': record.get('name'),
                    'phone': record.get('phone'),
                    'wechat': record.get('wechat_id'),
                    'source': f"external:{record.get('source', 'unknown')}"
                }
                self.entity_registry.register_entity(entity_attrs)

    async def _handle_entity_update(self, data: Dict[str, Any]):
        pass

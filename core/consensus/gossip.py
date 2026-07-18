"""
Gossip Protocol Implementation
===============================
Implements epidemic gossip protocol for distributed consensus.
"""

import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GossipMessage:
    """Gossip message structure."""

    sender: str
    recipients: List[str]
    timestamp: float
    payload: Dict[str, Any]
    message_id: str
    hop_count: int = 0
    ttl: int = 5


@dataclass
class GossipState:
    """Node state for gossip protocol."""

    node_id: str
    version: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)


class GossipProtocol:
    """
    Implements epidemic gossip for state propagation.
    """

    def __init__(
        self, node_id: str, fanout: int = 3, interval: float = 1.0, max_messages: int = 1000
    ):
        self.node_id = node_id
        self.fanout = fanout
        self.interval = interval
        self.max_messages = max_messages

        self.state = GossipState(node_id=node_id)
        self.peers: Set[str] = set()
        self.message_queue: List[GossipMessage] = []
        self.seen_messages: Set[str] = set()
        self.pending_updates: Dict[str, Any] = {}

        self.running = False
        self.gossip_task = None
        self.heartbeat_task = None

    async def start(self):
        """Start gossip protocol."""
        self.running = True
        self.gossip_task = asyncio.create_task(self._gossip_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop gossip protocol."""
        self.running = False
        if self.gossip_task:
            self.gossip_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

    def add_peer(self, peer_id: str):
        """Add peer to gossip network."""
        self.peers.add(peer_id)

    def remove_peer(self, peer_id: str):
        """Remove peer from gossip network."""
        self.peers.discard(peer_id)

    def update_state(self, key: str, value: Any):
        """Update local state."""
        self.state.data[key] = value
        self.state.version += 1
        self.state.last_update = time.time()
        self.pending_updates[key] = value

    async def send_gossip(self, recipient: str, message: GossipMessage):
        """Send gossip message to a peer."""
        # In practice, this would send over network
        # For simulation, we'll handle locally
        await self._receive_gossip(recipient, message)

    async def _gossip_loop(self):
        """Main gossip propagation loop."""
        while self.running:
            try:
                await asyncio.sleep(self.interval)

                if not self.peers or not self.pending_updates:
                    continue

                # Select random peers
                peers = random.sample(list(self.peers), min(self.fanout, len(self.peers)))

                # Create gossip message
                message = GossipMessage(
                    sender=self.node_id,
                    recipients=peers,
                    timestamp=time.time(),
                    payload={
                        "node_id": self.node_id,
                        "version": self.state.version,
                        "data": self.state.data.copy(),
                        "updates": self.pending_updates.copy(),
                    },
                    message_id=f"{self.node_id}_{int(time.time())}_{random.randint(0, 1000)}",
                )

                # Send to peers
                for peer in peers:
                    await self.send_gossip(peer, message)

                # Clear pending updates
                self.pending_updates.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Gossip loop error: {e}")

    async def _heartbeat_loop(self):
        """Heartbeat loop for peer discovery."""
        while self.running:
            try:
                await asyncio.sleep(self.interval * 10)

                # Send heartbeat to all peers
                heartbeat = GossipMessage(
                    sender=self.node_id,
                    recipients=list(self.peers),
                    timestamp=time.time(),
                    payload={"type": "heartbeat", "node_id": self.node_id},
                    message_id=f"heartbeat_{self.node_id}_{int(time.time())}",
                    ttl=1,
                )

                for peer in self.peers:
                    await self.send_gossip(peer, heartbeat)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat loop error: {e}")

    async def _receive_gossip(self, sender: str, message: GossipMessage):
        """Receive and process gossip message."""
        # Check if we've seen this message
        if message.message_id in self.seen_messages:
            return

        self.seen_messages.add(message.message_id)
        if len(self.seen_messages) > self.max_messages:
            # Clean old messages
            self.seen_messages = set(list(self.seen_messages)[-self.max_messages // 2 :])

        # Process message
        if message.payload.get("type") == "heartbeat":
            await self._process_heartbeat(message)
        else:
            await self._process_update(message)

    async def _process_heartbeat(self, message: GossipMessage):
        """Process heartbeat message."""
        sender = message.payload.get("node_id")
        if sender and sender != self.node_id:
            self.add_peer(sender)

    async def _process_update(self, message: GossipMessage):
        """Process state update message."""
        sender = message.sender

        # Check version
        remote_version = message.payload.get("version", 0)
        if remote_version <= self.state.version:
            return

        # Merge data
        remote_data = message.payload.get("data", {})
        self.state.data.update(remote_data)
        self.state.version = remote_version
        self.state.last_update = time.time()

        # Forward to peers (if TTL allows)
        if message.hop_count < message.ttl:
            message.hop_count += 1
            peers = random.sample(
                list(self.peers - {sender}), min(self.fanout, len(self.peers) - 1)
            )
            for peer in peers:
                await self.send_gossip(peer, message)

    def get_state(self, key: str = None) -> Any:
        """Get local state."""
        if key is None:
            return self.state.data
        return self.state.data.get(key)

    def get_statistics(self) -> Dict[str, Any]:
        """Get gossip protocol statistics."""
        return {
            "node_id": self.node_id,
            "peers": len(self.peers),
            "version": self.state.version,
            "pending_updates": len(self.pending_updates),
            "message_queue": len(self.message_queue),
            "seen_messages": len(self.seen_messages),
            "last_update": self.state.last_update,
        }


class GossipSimulator:
    """
    Simulator for testing gossip protocol.
    """

    def __init__(self, n_nodes: int = 10):
        self.n_nodes = n_nodes
        self.nodes = {}

        # Create nodes
        for i in range(n_nodes):
            node_id = f"node_{i}"
            self.nodes[node_id] = GossipProtocol(node_id)

        # Connect nodes
        for i, node in enumerate(self.nodes.values()):
            for j, other in enumerate(self.nodes.values()):
                if i != j:
                    node.add_peer(other.node_id)

    async def run_simulation(self, duration: int = 10):
        """
        Run gossip simulation for specified duration.
        """
        # Start all nodes
        tasks = []
        for node in self.nodes.values():
            tasks.append(node.start())

        # Wait for protocol to stabilize
        await asyncio.sleep(1)

        # Generate updates
        for i, node in enumerate(self.nodes.values()):
            node.update_state(f"value_{i}", random.random())

        # Run for specified duration
        await asyncio.sleep(duration)

        # Stop all nodes
        for node in self.nodes.values():
            await node.stop()

        # Get final statistics
        stats = {}
        for node_id, node in self.nodes.items():
            stats[node_id] = node.get_statistics()

        return stats

    def get_consensus_quality(self) -> float:
        """
        Check quality of consensus achieved.
        """
        values = []
        for node in self.nodes.values():
            state = node.get_state()
            if state:
                # Extract numeric values
                values.extend([v for v in state.values() if isinstance(v, (int, float))])

        if not values:
            return 0.0

        # Compute variance across nodes
        mean = np.mean(values)
        variance = np.var(values)
        return 1.0 / (1.0 + variance)

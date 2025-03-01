# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue
import uuid
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, List, Optional, Set, Type, Union

import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.utils import BackgroundProcHandle

logger = init_logger(__name__)

AnyFuture = Union[asyncio.Future[Any], Future[Any]]


class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing 
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":

        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported.")

        if multiprocess_mode and asyncio_mode:
            return AsyncMPClient(vllm_config, executor_class, log_stats)

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @abstractmethod
    def shutdown(self):
        ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True) -> None:
        raise NotImplementedError

    def reset_prefix_cache(self) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    def wake_up(self) -> None:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: List[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(self, is_start: bool = True) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(self) -> None:
        raise NotImplementedError

    async def sleep_async(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up_async(self) -> None:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: List[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> Set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended 
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        return self.engine_core.step()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True) -> None:
        self.engine_core.profile(is_start)

    def reset_prefix_cache(self) -> None:
        self.engine_core.reset_prefix_cache()

    def sleep(self, level: int = 1) -> None:
        self.engine_core.sleep(level)

    def wake_up(self) -> None:
        self.engine_core.wake_up()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: Union[zmq.Context, zmq.asyncio.Context] = None
    output_socket: Union[zmq.Socket, zmq.asyncio.Socket] = None
    input_socket: Union[zmq.Socket, zmq.asyncio.Socket] = None
    proc_handle: Optional[BackgroundProcHandle] = None

    def __call__(self):
        """Clean up background resources."""

        if self.proc_handle is not None:
            self.proc_handle.shutdown()
        # ZMQ context termination can hang if the sockets
        # aren't explicitly closed first.
        if self.output_socket is not None:
            self.output_socket.close(linger=0)
        if self.input_socket is not None:
            self.input_socket.close(linger=0)
        if self.ctx is not None:
            self.ctx.destroy(linger=0)


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket
    
        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool,
    ):
        # Serialization setup.
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # ZMQ setup.
        self.ctx = (
            zmq.asyncio.Context()  # type: ignore[attr-defined]
            if asyncio_mode else zmq.Context())  # type: ignore[attr-defined]

        # This will ensure resources created so far are closed
        # when the client is garbage collected,  even if an
        # exception is raised mid-construction.
        resources = BackgroundResources(ctx=self.ctx)
        self._finalizer = weakref.finalize(self, resources)

        # Paths and sockets for IPC.
        output_path = get_open_zmq_ipc_path()
        input_path = get_open_zmq_ipc_path()
        resources.output_socket = make_zmq_socket(self.ctx, output_path,
                                                  zmq.constants.PULL)
        resources.input_socket = make_zmq_socket(self.ctx, input_path,
                                                 zmq.constants.PUSH)

        # Start EngineCore in background process.
        self.is_engine_dead = False
        resources.proc_handle = BackgroundProcHandle(
            input_path=input_path,
            output_path=output_path,
            process_name="EngineCore",
            target_fn=EngineCoreProc.run_engine_core,
            process_kwargs={
                "vllm_config": vllm_config,
                "executor_class": executor_class,
                "log_stats": log_stats,
            })
        resources.proc_handle.wait_for_startup()

        self.output_socket = resources.output_socket
        self.input_socket = resources.input_socket
        self.utility_results: Dict[int, AnyFuture] = {}

    def shutdown(self):
        self._finalizer()

    def _raise_if_engine_core_dead(self, buffer: Any):
        if buffer == EngineCoreProc.ENGINE_CORE_DEAD:
            self.is_engine_dead = True
            raise EngineDeadError()

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""

        return (EngineDeadError(
            suppress_context=True) if self.is_engine_dead else e)


def _process_utility_output(output: UtilityOutput,
                            utility_results: Dict[int, AnyFuture]):
    """Set the result from a utility method in the waiting future"""
    future = utility_results.pop(output.call_id)
    if output.failure_message is not None:
        future.set_exception(Exception(output.failure_message))
    else:
        future.set_result(output.result)


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: Type[Executor],
                 log_stats: bool):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.outputs_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        output_socket = self.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        def process_outputs_socket():
            try:
                while True:
                    (frame, ) = output_socket.recv_multipart(copy=False)
                    outputs = decoder.decode(frame.buffer)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output,
                                                utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except zmq.error.ContextTerminated:
                # Expected when the class is GC'd / during process termination.
                pass

        # Process outputs from engine in separate thread.
        Thread(target=process_outputs_socket, daemon=True).start()

    def get_output(self) -> EngineCoreOutputs:
        try:
            (frame, ) = self.output_socket.recv_multipart(copy=False)
            self._raise_if_engine_core_dead(frame.buffer)
            return self.outputs_queue.get()
        except Exception as e:
            raise self._format_exception(e) from None

    def _send_input(self, request_type: EngineCoreRequestType,
                    request: Any) -> None:
        try:
            # (RequestType, SerializedRequest)
            msg = (request_type.value, self.encoder.encode(request))
            self.input_socket.send_multipart(msg, copy=False)
        except Exception as e:
            raise self._format_exception(e) from None

    def _call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future

        self._send_input(EngineCoreRequestType.UTILITY,
                         (call_id, method, args))

        return future.result()

    def add_request(self, request: EngineCoreRequest) -> None:
        # NOTE: text prompt is not needed in the core engine as it has been
        # tokenized.
        request.prompt = None
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True) -> None:
        self._call_utility("profile", is_start)

    def reset_prefix_cache(self) -> None:
        self._call_utility("reset_prefix_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self._call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self._call_utility("remove_lora", lora_id)

    def list_loras(self) -> Set[int]:
        return self._call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self._call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1) -> None:
        self._call_utility("sleep", level)

    def wake_up(self) -> None:
        self._call_utility("wake_up")

    def execute_dummy_batch(self) -> None:
        self._call_utility("execute_dummy_batch")


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: Type[Executor],
                 log_stats: bool):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        self.outputs_queue: asyncio.Queue[Union[EngineCoreOutputs,
                                                Exception]] = asyncio.Queue()
        self.process_outputs_socket_task: Optional[asyncio.Task] = None

    def shutdown(self):
        super().shutdown()
        if task := getattr(self, "process_outputs_socket_task", None):
            task.cancel()

    async def _start_process_outputs_socket(self):
        # Perform IO (releases GIL) in background task.
        # Avoid task having direct reference back to the client.
        output_socket = self.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        async def process_outputs_socket():
            try:
                while True:
                    (frame, ) = await output_socket.recv_multipart(copy=False)
                    self._raise_if_engine_core_dead(frame.buffer)
                    outputs: EngineCoreOutputs = decoder.decode(frame.buffer)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output,
                                                utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)

        self.process_outputs_socket_task = asyncio.create_task(
            process_outputs_socket())

    async def get_output_async(self) -> EngineCoreOutputs:
        if self.process_outputs_socket_task is None:
            await self._start_process_outputs_socket()

        # Exceptions in process_outputs_socket are forwarded
        # so we can raise them in the run_output_handler() task.
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None

        return self.decoder.decode(outputs)

    async def _send_input(self, request_type: EngineCoreRequestType,
                          request: Any) -> None:
        try:
            msg = (request_type.value, self.encoder.encode(request))
            await self.input_socket.send_multipart(msg, copy=False)
        except Exception as e:
            raise self._format_exception(e) from None

    async def _call_utility_async(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        await self._send_input(EngineCoreRequestType.UTILITY,
                               (call_id, method, args))

        return await future

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # NOTE: text prompt is not needed in the core engine as it has been
        # tokenized.
        request.prompt = None
        await self._send_input(EngineCoreRequestType.ADD, request)

    async def abort_requests_async(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile_async(self, is_start: bool = True) -> None:
        await self._call_utility_async("profile", is_start)

    async def reset_prefix_cache_async(self) -> None:
        await self._call_utility_async("reset_prefix_cache")

    async def sleep_async(self, level: int = 1) -> None:
        await self._call_utility_async("sleep", level)

    async def wake_up_async(self) -> None:
        await self._call_utility_async("wake_up")

    async def execute_dummy_batch_async(self) -> None:
        await self._call_utility_async("execute_dummy_batch")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self._call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self._call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> Set[int]:
        return await self._call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self._call_utility_async("pin_lora", lora_id)

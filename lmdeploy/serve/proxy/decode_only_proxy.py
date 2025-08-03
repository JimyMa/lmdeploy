# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import copy
import json
import os
import os.path as osp
import random
import threading
import time
from collections import deque
from http import HTTPStatus
from typing import Deque, Dict, List, Literal, Optional, Union

import aiohttp
import numpy as np
import requests
import uvicorn
import logging
import sys
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, EngineRole, RDMALinkType, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.serve.openai.api_server import check_api_key, create_error_response
from lmdeploy.serve.openai.protocol import ModelCard  # noqa: E501
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest, ModelList, ModelPermission
from lmdeploy.serve.proxy.constants import AIOHTTP_TIMEOUT, LATENCY_DEQUE_LEN, ErrorCodes, RoutingStrategy, err_msg

from viztracer import VizTracer

log_folder = "/nvme4/share/chenjiefei/scripts/proxy_log/"

if not os.path.exists(log_folder):
    os.makedirs(log_folder, exist_ok=True)

# 获取包含详细时间的时间戳（年-月-日-时-分-秒）
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
# 定义包含详细时间的日志文件名
log_filename = os.path.join(log_folder, f"proxy_res_{current_datetime}.log")

print(f"日志文件将保存到: {log_filename}")

logger = logging.getLogger('proxy')
logger.setLevel(logging.INFO)

if logger.handlers:
    logger.handlers.clear()

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

history_itl_count = 500

class Status(BaseModel):
    """Status protocol consists of models' information."""
    role: EngineRole = EngineRole.Hybrid
    models: Optional[List[str]] = Field(default=[], examples=[[]])
    unfinished: int = 0
    latency: Deque = Field(default=deque(maxlen=LATENCY_DEQUE_LEN), examples=[[]])
    speed: Optional[int] = Field(default=None, examples=[None])
    kvcache_usage: Optional[float] = 0
    num_running: Optional[int] = 0
    num_waiting: Optional[int] = 0
    total_token_nums: Optional[int] = 0
    history_itl: Deque[int] = deque(maxlen=history_itl_count)

class Node(BaseModel):
    """Node protocol consists of url and status."""
    url: str
    status: Optional[Status] = None


CONTROLLER_HEART_BEAT_EXPIRATION = int(os.getenv('LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION', 90))


def heart_beat_controller(proxy_controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        logger.info('Start heart beat check')
        proxy_controller.remove_stale_nodes_by_expiration()


class NodeManager:
    """Manage all the sub nodes.

    Args:
        config_path (str): the path of the config file.
        strategy (str): the strategy to dispatch node to handle the requests.
            - random: not fully radom, but decided by the speed of nodes.
            - min_expected_latency: will compute the expected latency to
                process the requests. The sooner of the node, the more requests
                will be dispatched to it.
            - min_observed_latency: Based on previous finished requests. The
                sooner they get processed, the more requests will be dispatched
                to.
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 serving_strategy: str = 'Hybrid',
                 routing_strategy: str = 'min_expected_latency',
                 migration_protocol: str = 'RDMA',
                 link_type: str = 'RoCE',
                 with_gdr: bool = True,
                 cache_status: Optional[bool] = True) -> None:
        self.nodes = dict()
        self.serving_strategy = ServingStrategy[serving_strategy]
        self.routing_strategy = RoutingStrategy.from_str(routing_strategy)

        self.cache_status = cache_status
        self.latencies = dict()
        self.config_path = osp.join(osp.dirname(osp.realpath(__file__)), 'proxy_config.json')
        print(f"config_path: {self.config_path}")
        if config_path is not None:
            self.config_path = config_path
        if osp.exists(self.config_path) and self.cache_status:
            with open(self.config_path, 'r') as config_file:
                if os.path.getsize(self.config_path) > 0:
                    logger.info(f'loading node configuration: {self.config_path}')
                    config = json.load(config_file)
                    # print(f"config: {config}")
                    self.nodes = {
                        node_url: Status.model_validate_json(node_status)
                        for node_url, node_status in config.items()
                    }
            # 初始化 self.node_locks，为每个节点创建一个锁
            self.node_locks = {node_url: threading.Lock() for node_url in self.nodes}
        else:
            # 如果不存在配置文件或者 cache_status 为 False，初始化为空字典
            self.node_locks = {}
 
        self.heart_beat_thread = threading.Thread(target=heart_beat_controller, args=(self, ), daemon=True)
        self.heart_beat_thread.start()
        self.aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)

        # For PD Disaggregation
        self.migration_protocol = MigrationProtocol[migration_protocol]
        self.rdma_config = DistServeRDMAConfig(with_gdr=with_gdr, link_type=RDMALinkType[link_type])
        self.pd_connection_pool = PDConnectionPool()
        self.dummy_prefill = False

        # For round_robin strategy
        self.prefill_index = 0
        self.decode_index = 0

        # For KV Cache load balance strategies, default collect decode kv cache usage per 3 seconds
        self.metric_collection_interval = 1
        self.metric_endpoint = '/proxy/metrics'
        self.log_interval = 1

        print(f'routing_strategy: {self.routing_strategy}')

        # 添加异步事件循环相关属性
        self.metric_loop = None
        self.metric_collector_task = None
        self.metric_running = False

        # 启动异步收集线程
        self.metric_running = True
        self.metric_collector_thread = threading.Thread(target=self._run_metric_collector, daemon=True)
        self.metric_collector_thread.start()

        self.metric_logger_thread = threading.Thread(target=self._log_metrics_loop, daemon=True)
        self.metric_logger_thread.start()

        # Profiler
        self.tracer = VizTracer(tracer_entries=10000000)

        self.connector = None
        self.session_pool = None
        self.pool_lock = asyncio.Lock()

    def get_nodes(self, role: EngineRole) -> Dict:
        items = list(self.nodes.items())
        return {node_url: node_status for (node_url, node_status) in items if node_status.role == role}

    @property
    def hybrid_nodes(self):
        return self.get_nodes(EngineRole.Hybrid)

    @property
    def prefill_nodes(self):
        return self.get_nodes(EngineRole.Prefill)

    @property
    def decode_nodes(self):
        return self.get_nodes(EngineRole.Decode)

    def _run_metric_collector(self):
        """在新线程中运行异步事件循环，收集所有指标"""
        self.metric_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.metric_loop)

        # 创建后台任务，重命名为更通用的指标收集任务
        self.metric_collector_task = self.metric_loop.create_task(self._collect_metrics_loop())

        # 运行事件循环
        self.metric_loop.run_forever()

        # 清理
        if self.metric_collector_task:
            self.metric_collector_task.cancel()
            try:
                self.metric_loop.run_until_complete(self.metric_collector_task)
            except asyncio.CancelledError:
                pass
        self.metric_loop.close()

    async def _collect_metrics_loop(self):
        """异步版本的指标收集循环，收集所有类型的指标"""
        while self.metric_running:
            await self._collect_metrics()
            await asyncio.sleep(self.metric_collection_interval)

    async def _collect_metrics(self):
        """异步并行收集所有Decode节点的指标（包括kvcache_usage和num_running）"""

        async def fetch_node(session, node_url):
            try:
                endpoint = f"{node_url.rstrip('/')}{self.metric_endpoint}"
                async with session.get(endpoint, timeout=2) as response:
                    if response.status == 200:
                        data = await response.json()
                        return node_url, data
            except Exception as e:
                logger.warning(f'Failed to collect metrics from {node_url}: {str(e)}')
            return node_url, None

        # 获取当前所有解码节点
        decode_nodes = list(self.decode_nodes.keys())
        if not decode_nodes:
            return

        # 使用连接池复用连接
        async with aiohttp.ClientSession(timeout=self.aiotimeout) as session:
            tasks = [fetch_node(session, node_url) for node_url in decode_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            for node_url, metrics in results:
                if metrics is not None:
                    status = self.nodes.get(node_url, Status())
                    # 更新kvcache_usage和num_running两个指标
                    status.kvcache_usage = metrics.get('kvcache_usage', 0)
                    status.num_waiting = metrics.get('num_waiting', 0)
                    status.num_running = metrics.get('num_running', 0)
                    self.nodes[node_url] = status

    def shutdown(self):
        """关闭所有后台线程"""
        # 停止指标收集器
        if self.metric_running:
            self.metric_running = False
            if self.metric_loop:
                self.metric_loop.call_soon_threadsafe(self.metric_loop.stop)
            if self.metric_collector_thread:
                self.metric_collector_thread.join(timeout=1.0)

    def _log_metrics_loop(self):
        """背景线程，定期收集指标并打印日志."""
        while True:
            time.sleep(self.log_interval)
            self._log_metrics()

    def _log_metrics(self, print_kv_cache: bool = True, print_batch_size: bool = False, 
                    print_avg_tpot: bool = False, print_num_running: bool = True, print_num_waiting: bool = True):
        """打印节点指标日志（增加num_running指标的打印）"""
        # 若所有参数均为 False，则不打印任何内容
        if not any([print_kv_cache, print_batch_size, print_avg_tpot, print_num_running]):
            return

        # 获取所有解码节点
        decode_nodes = self.decode_nodes
        if not decode_nodes:
            logger.info('No decode nodes available')
            return

        # # 检查是否存在至少一个节点的batch size >= 1
        # has_active_node = any((status.unfinished or 0) >= 1 for _, status in decode_nodes.items())

        # # 如果不存在任何活跃节点，则不打印任何节点的日志
        # if not has_active_node:
        #     return

        # 根据参数组合生成日志标题
        title_parts = []
        if print_kv_cache:
            title_parts.append('KV Cache')
        if print_batch_size:
            title_parts.append('Batch Size')
        if print_avg_tpot:
            title_parts.append('Avg TPOT')
        if print_num_running:
            title_parts.append('Running Requests')

        logger.info(f"=== {' & '.join(title_parts)} Metrics ===")

        # 遍历所有节点并输出指标
        for index, (node_url, status) in enumerate(decode_nodes.items(), start=1):
            log_items = [f'Node: {index}']

            if print_kv_cache:
                usage = status.kvcache_usage or 0
                total_tokens = status.total_token_nums or 0
                log_items.append(f'KV Cache Usage: {usage:.2f}, Total Tokens: {total_tokens}')

            if print_num_running:
                num_running = status.num_running or 0
                log_items.append(f'Running Requests: {num_running}')

            if print_num_waiting:
                num_waiting = status.num_waiting or 0
                log_items.append(f'Waiting Requests: {num_waiting}')

            if print_batch_size:
                batch_size = status.unfinished or 0
                log_items.append(f'Batch Size: {batch_size}')

            if print_avg_tpot:
                # 计算平均 history_itl (Tokens Per Output Token)
                avg_history_itl = sum(status.history_itl) / len(status.history_itl) if status.history_itl else 0
                log_items.append(f'Avg TPOT: {avg_history_itl:.2f}')

            logger.info(', '.join(log_items))

    def update_config_file(self):
        """Update the config file."""
        nodes = copy.deepcopy(self.nodes)
        for _, status in nodes.items():
            status.latency = deque(list(status.latency)[-LATENCY_DEQUE_LEN:])
        if self.cache_status:
            with open(self.config_path, 'w') as config_file:  # update cfg yml
                json.dump({
                    node_url: node_status.model_dump_json()
                    for node_url, node_status in nodes.items()
                },
                          config_file,
                          indent=2)

    def add(self, node_url: str, status: Optional[Status] = None):
        """Add a node to the manager.

        Args:
            node_url (str): A http url. Can be the url generated by
                `lmdeploy serve api_server`.
            description (Dict): The description of the node. An example:
                {'http://0.0.0.0:23333': {models: ['internlm-chat-7b]},
                speed: -1}. The speed here can be RPM or other metric. All the
                values of nodes should be the same metric.
        """
        if status is None:
            status = self.nodes.get(node_url, Status())
        if status.models != []:  # force register directly
            self.remove(node_url)
            self.nodes[node_url] = status
            self.update_config_file()
            # 初始化节点的锁
            self.node_locks[node_url] = threading.Lock()
            return
        try:
            from lmdeploy.serve.openai.api_client import APIClient
            client = APIClient(api_server_url=node_url)
            status.models = client.available_models
            self.nodes[node_url] = status
            # 初始化节点的锁
            self.node_locks[node_url] = threading.Lock()
        except requests.exceptions.RequestException as e:  # noqa
            logger.error(f'exception happened when adding node {node_url}, {e}')
            return self.handle_api_timeout(node_url)
        self.update_config_file()

    def remove(self, node_url: str):
        """Remove a node."""
        if node_url in self.nodes.keys():
            self.nodes.pop(node_url)
            self.update_config_file()
            dropped_conn = []
            for conn in self.pd_connection_pool.pool:
                if node_url in conn:
                    dropped_conn.append(conn)
            for conn in dropped_conn:
                self.pd_connection_pool.drop(*conn)

        # 清理锁
        if node_url in self.node_locks:
            del self.node_locks[node_url]

    def terminate_node(self, node_url: str):
        """Terminate a node."""
        success = True
        if node_url in self.nodes:
            self.nodes.pop(node_url)
            headers = {'accept': 'application/json'}
            try:
                response = requests.get(f'{node_url}/terminate', headers=headers)
                if response.status_code != 200:
                    success = False
                    logger.error(f'Failed to terminate node {node_url}, '
                                 f'error_code={response.status_code}, '
                                 f'error_msg={response.text}')
            except Exception as e:  # noqa
                logger.error(f'exception happened when terminating node {node_url}, {e}')
                success = False
        else:
            logger.error(f'terminating node {node_url} failed since it does not exist. '
                         'May try /nodes/status to check the node list')
            success = False
        self.update_config_file()
        # 清理锁
        if node_url in self.node_locks:
            del self.node_locks[node_url]
        return success

    def terminate_all_nodes(self):
        """Terminate all nodes."""
        node_url_li = list(self.nodes.keys())
        all_success = True
        for node_url in node_url_li:
            if not self.terminate_node(node_url):
                all_success = False
        return all_success

    def remove_stale_nodes_by_expiration(self):
        """Remove stale nodes."""
        to_be_deleted = []
        node_urls = list(self.nodes.keys())
        for node_url in node_urls:
            url = f'{node_url}/health'
            headers = {'accept': 'application/json'}
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    to_be_deleted.append(node_url)
            except:  # noqa
                to_be_deleted.append(node_url)
        for node_url in to_be_deleted:
            self.remove(node_url)
            logger.info(f'Removed node_url: {node_url} '
                        'due to heart beat expiration')

    @property
    def model_list(self):
        """Supported model list."""
        model_names = []
        items = list(self.nodes.items())
        for _, status in items:
            model_names.extend(status.models)
        return model_names

    @property
    def status(self):
        """Return the status."""
        return self.nodes

    def get_node_url(self, model_name: str, role: EngineRole = EngineRole.Hybrid, prefill_length: Optional[int] = 0):
        """Add a node to the manager.

        Args:
            model_name (str): A http url. Can be the url generated by
                `lmdeploy serve api_server`.
        Return:
            A node url or None.
        """

        def get_matched_urls():
            urls_with_speeds, speeds, urls_without_speeds = [], [], []
            for node_url, status in self.get_nodes(role).items():
                if model_name in status.models:
                    if status.speed is not None:
                        urls_with_speeds.append(node_url)
                        speeds.append(status.speed)
                    else:
                        urls_without_speeds.append(node_url)
            all_matched_urls = urls_with_speeds + urls_without_speeds
            if len(all_matched_urls) == 0:
                return None
            # some nodes does not contain speed
            # we can set them the average speed value
            average_speed = sum(speeds) / len(speeds) if len(speeds) else 1
            all_the_speeds = speeds + [average_speed] * len(urls_without_speeds)
            return all_matched_urls, all_the_speeds
        
        def update_token_count(selected_url):
            """更新选中节点的token计数."""
            if selected_url and prefill_length > 0:
                node_lock = self.node_locks.get(selected_url, threading.Lock())
                with node_lock:
                    if selected_url in self.nodes:
                        self.nodes[selected_url].total_token_nums += prefill_length

        if self.routing_strategy == RoutingStrategy.RANDOM:
            all_matched_urls, all_the_speeds = get_matched_urls()
            if len(all_matched_urls) == 0:
                return None
            speed_sum = sum(all_the_speeds)
            weights = [speed / speed_sum for speed in all_the_speeds]
            index = random.choices(range(len(all_matched_urls)), weights=weights)[0]
            url = all_matched_urls[index]
            update_token_count(url)
            return url
        elif self.routing_strategy == RoutingStrategy.MIN_EXPECTED_LATENCY:
            all_matched_urls, all_the_speeds = get_matched_urls()
            if len(all_matched_urls) == 0:
                return None
            min_latency = float('inf')
            min_index = 0
            # random traverse nodes for low concurrency situation
            all_indexes = [i for i in range(len(all_the_speeds))]
            random.shuffle(all_indexes)
            for index in all_indexes:
                latency = self.get_nodes(role)[all_matched_urls[index]].unfinished / all_the_speeds[index]
                if min_latency > latency:
                    min_latency = latency
                    min_index = index
            url = all_matched_urls[min_index]
            update_token_count(url)
            return url
        elif self.routing_strategy == RoutingStrategy.MIN_OBSERVED_LATENCY:
            all_matched_urls, latencies = [], []
            for node_url, node_status in self.get_nodes(role).items():
                if model_name in node_status.models:
                    if len(node_status.latency):
                        latencies.append(np.mean(np.array(node_status.latency)))
                    else:
                        latencies.append(float('inf'))
                    all_matched_urls.append(node_url)
            if len(all_matched_urls) == 0:
                return None
            index = np.argmin(np.array(latencies))
            url = all_matched_urls[index]
            update_token_count(url)
            return url
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            matched_nodes = [node_url for node_url, status in self.get_nodes(role).items()]
            if not matched_nodes:
                print('no matched nodes')
                return None
            if role == EngineRole.Prefill:
                current_index = self.prefill_index
                self.prefill_index = (self.prefill_index + 1) % len(matched_nodes)
            elif role == EngineRole.Decode:
                current_index = self.decode_index
                self.decode_index = (self.decode_index + 1) % len(matched_nodes)
            else:
                current_index = 0
            current_index = current_index % len(matched_nodes)
            print(f'round robin index: {current_index}')
            url = matched_nodes[current_index]
            update_token_count(url)
            return url
        elif self.routing_strategy == RoutingStrategy.BATCH_SIZE_BALANCE:
            matched_nodes = [node_url for node_url, status in self.get_nodes(role).items()]
            if not matched_nodes:
                return None
            # 找到 unfinished 最小的节点
            min_unfinished = float('inf')
            selected_url = None
            for node_url in matched_nodes:
                unfinished = self.nodes[node_url].unfinished
                if unfinished < min_unfinished:
                    min_unfinished = unfinished
                    selected_url = node_url
            update_token_count(selected_url)
            return selected_url
        elif self.routing_strategy == RoutingStrategy.KV_CACHE_BALANCE:
            if role == EngineRole.Decode:
                # 使用总token数作为负载均衡依据
                min_total_tokens = float('inf')
                selected_url = None

                # 收集所有符合条件的节点及其token数
                all_node_tokens = {}
                node_index = 0

                for node_url, status in self.decode_nodes.items():
                    if model_name in status.models:
                        # 使用独立的锁保护读取操作
                        with self.node_locks.get(node_url, threading.Lock()):
                            total_tokens = status.total_token_nums

                        # 存储为 node 0, node 1, ... 的形式
                        all_node_tokens[f'node {node_index}'] = total_tokens

                        if total_tokens < min_total_tokens:
                            min_total_tokens = total_tokens
                            selected_url = node_url

                        node_index += 1

                update_token_count(selected_url)

                # print(f"all_node_tokens: {all_node_tokens}")

                return selected_url

            elif role == EngineRole.Prefill:
                # Prefill节点使用Round Robin策略
                matched_nodes = [
                    node_url for node_url, status in self.get_nodes(role).items() if model_name in status.models
                ]
                if not matched_nodes:
                    return None
                current_index = self.prefill_index
                self.prefill_index = (self.prefill_index + 1) % len(matched_nodes)
                url = matched_nodes[current_index]
                update_token_count(url)
                return url

            # 对于Hybrid或其他类型节点，使用随机策略作为后备
            all_matched_urls, all_the_speeds = get_matched_urls()
            if all_matched_urls is None:
                return None
            speed_sum = sum(all_the_speeds)
            weights = [speed / speed_sum for speed in all_the_speeds]
            index = random.choices(range(len(all_matched_urls)), weights=weights)[0]
            url = all_matched_urls[index]
            update_token_count(url)
            return url
        else:
            raise ValueError(f'Invalid strategy: {self.routing_strategy}')

    async def check_request_model(self, model_name) -> Optional[JSONResponse]:
        """Check if a request is valid."""
        if model_name in self.model_list:
            return
        ret = create_error_response(HTTPStatus.NOT_FOUND, f'The model `{model_name}` does not exist.')
        return ret

    def handle_unavailable_model(self, model_name):
        """Handle unavailable model.

        Args:
            model_name (str): the model in the request.
        """
        logger.warning(f'no model name: {model_name}')
        ret = {
            'error_code': ErrorCodes.MODEL_NOT_FOUND,
            'text': err_msg[ErrorCodes.MODEL_NOT_FOUND],
        }
        return json.dumps(ret).encode() + b'\n'

    def handle_api_timeout(self, node_url):
        """Handle the api time out."""
        logger.warning(f'api timeout: {node_url}')
        ret = {
            'error_code': ErrorCodes.API_TIMEOUT.value,
            'text': err_msg[ErrorCodes.API_TIMEOUT],
        }
        return json.dumps(ret).encode() + b'\n'

    async def get_session(self):
        """异步获取或创建连接池会话"""
        if self.session_pool is None:
            async with self.pool_lock:
                if self.session_pool is None:  # 双重检查
                    self.connector = aiohttp.TCPConnector(limit=16384,limit_per_host=16384)
                    self.session_pool = aiohttp.ClientSession(
                        connector=self.connector,
                        timeout=self.aiotimeout
                    )

        return self.session_pool

    async def stream_generate(self, request: Dict, node_url: str, endpoint: str, proxy_recv_time: Optional[float] = None):
        """Return a generator to handle the input request.

        Args:
            request (Dict): the input request.
            node_url (str): the node url.
            endpoint (str): the endpoint. Such as `/v1/chat/completions`.
        """
        last_tokens = 0  # 记录该请求的token数
        last_completion_tokens = 0  # 记录该请求的completion_tokens数
        last_decode_time = None  # 记录上次Decode的时间
        queueing_latency = 0
        try:
            request['stream_options'] = {'include_usage': True}  # 添加stream_options字段，用于统计Decode的KV Cache使用量
            # 获取连接池中的会话
            # session = await self.get_session()
            # async with session.post(node_url + endpoint, json=request, timeout=self.aiotimeout) as response:
            async with aiohttp.ClientSession() as session:
                async with session.post(node_url + endpoint, json=request, timeout=self.aiotimeout) as response:
                    async for line in response.content:
                        # print(f"stream output line: {line}")
                        # 解析token使用情况
                        if line.startswith(b'data: '):
                            line_str = line.decode('utf-8').strip()
                            if line_str != 'data: [DONE]':
                                try:
                                    data = json.loads(line[len(b'data: '):])
                                    
                                    # 添加proxy相关时间字段
                                    # 确保usage字段存在
                                    if 'usage' not in data:
                                        data['usage'] = {}

                                    # 添加proxy_send_time字段
                                    proxy_send_time = time.time()
                                    data['usage']['proxy_send_time'] = proxy_send_time
                                    
                                    # 当proxy_recv_time不为None时添加该字段
                                    if proxy_recv_time is not None:
                                        data['usage']['proxy_recv_time'] = proxy_recv_time
                                    
                                    # 检查是否有usage字段并更新
                                    if 'total_tokens' in data['usage']:
                                        current_completion_tokens = data['usage']['completion_tokens']
                                        # 计算token增量
                                        token_delta = current_completion_tokens - last_completion_tokens
                                        last_completion_tokens = current_completion_tokens
                                        last_tokens = data['usage']['total_tokens']

                                        # 更新节点总token数
                                        if node_url in self.nodes:
                                            with self.node_locks.get(node_url, threading.Lock()):
                                                self.nodes[node_url].total_token_nums += token_delta

                                        # 记录Decode阶段的token间延迟
                                        if current_completion_tokens > 1:  # 跳过Prefill阶段
                                            current_time = time.time()
                                            if last_decode_time is not None:
                                                itl = int((current_time - last_decode_time) * 1000)  # 转换为毫秒
                                                if node_url in self.nodes:
                                                    self.nodes[node_url].history_itl.append(itl)
                                            last_decode_time = current_time
                                        elif current_completion_tokens == 1:  # 第一个Decode token，初始化时间
                                            last_decode_time = time.time()

                                    if 'queued_time' in data['usage']:
                                        queueing_latency = data['usage']['queued_time']
                                    
                                    # 将修改后的数据转换回字节
                                    modified_line = b'data: ' + json.dumps(data).encode('utf-8')
                                except json.JSONDecodeError:
                                    logger.warning(f'Failed to parse JSON: {line}')
                                    modified_line = line  # 解析失败时使用原始行
                            else:
                                modified_line = line  # 对于[DONE]消息不做修改
                        else:
                            modified_line = line  # 非data行不做修改

                        if modified_line.strip():
                            yield modified_line + b'\n\n'

        except (Exception, GeneratorExit, aiohttp.ClientError) as e:  # noqa
            logger.error(f'catched an exception: {e}')
            # exception happened, reduce unfinished num
            yield self.handle_api_timeout(node_url)

        finally:
            # 请求结束时减去该请求的token数
            if last_tokens > 0:
                with self.node_locks.get(node_url, threading.Lock()):
                    if node_url in self.nodes:
                        self.nodes[node_url].total_token_nums -= last_tokens

            # with self.node_locks.get(node_url, threading.Lock()):
            #     if node_url in self.nodes:
            #         self.nodes[node_url].total_token_nums -= (request["input_ids"][0] + request["max_tokens"])

            print(f"queueing_latency: {queueing_latency:.4f} s")

    async def generate(self, request: Dict, node_url: str, endpoint: str):
        """Return a the response of the input request.

        Args:
            request (Dict): the input request.
            node_url (str): the node url.
            endpoint (str): the endpoint. Such as `/v1/chat/completions`.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(node_url + endpoint, json=request, timeout=self.aiotimeout) as response:
                    return await response.text()
        except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:  # noqa  # yapf: disable
            logger.error(f'catched an exception: {e}')
            return self.handle_api_timeout(node_url)

    def pre_call(self, node_url):
        """Preprocess before the request get processed.

        Args:
            node_url (str): the node url.
        """
        self.nodes[node_url].unfinished += 1
        return time.time()

    def post_call(self, node_url: str, start: int):
        """Post process after the response finished.

        Args:
            node_url (str): the node url.
            start (int): the start time point. time.time()
        """
        self.nodes[node_url].unfinished -= 1
        self.nodes[node_url].latency.append(time.time() - start)

    def create_background_tasks(self, url: str, start: int):
        """To create a background task.

        Args:
            node_url (str): the node url.
            start (int): the start time point. time.time()
        """
        background_tasks = BackgroundTasks()
        background_tasks.add_task(self.post_call, url, start)
        return background_tasks


app = FastAPI(docs_url='/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
node_manager: NodeManager = None

# curl -X POST "http://0.0.0.0:8050/profiler/start"
@app.post('/profiler/start', dependencies=[Depends(check_api_key)])
def start_profiler():
    """启动性能分析"""
    node_manager.tracer.start()
    return {"status": "Profiler start"}

# curl -X POST "http://0.0.0.0:8050/profiler/stop"
@app.post('/profiler/stop', dependencies=[Depends(check_api_key)])
def stop_profiler(output_file: Optional[str] = f"/nvme4/share/chenjiefei/scripts/proxy_trace/proxy_{time.time():.6f}.json"):
    node_manager.tracer.stop()
    # 保存结果，支持自定义输出文件
    if output_file:
        node_manager.tracer.save(output_file=output_file)
    else:
        node_manager.tracer.save()

    # 打印保存信息
    logger.info(f"Profiler results saved successfully. File: {output_file}")

    return {"status": "Profiler stopped and results saved"}

@app.get('/v1/models', dependencies=[Depends(check_api_key)])
def available_models():
    """Show available models."""
    model_cards = []
    for model_name in node_manager.model_list:
        model_cards.append(ModelCard(id=model_name, root=model_name, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.get('/nodes/status', dependencies=[Depends(check_api_key)])
def node_status():
    """Show nodes status."""
    try:
        return node_manager.status
    except:  # noqa
        return False


@app.post('/nodes/add', dependencies=[Depends(check_api_key)])
def add_node(node: Node, raw_request: Request = None):
    """Add a node to the manager.

    - url (str): A http url. Can be the url generated by
        `lmdeploy serve api_server`.
    - status (Dict): The description of the node. An example:
        {models: ['internlm-chat-7b],  speed: 1}. The speed here can be
        RPM or other metric. All the values of nodes should be the same metric.
    """
    try:
        res = node_manager.add(node.url, node.status)
        if res is not None:
            logger.error(f'add node {node.url} failed, {res}')
            return res
        logger.info(f'add node {node.url} successfully')
        return 'Added successfully'
    except:  # noqa
        return 'Failed to add, please check the input url.'


@app.post('/nodes/remove', dependencies=[Depends(check_api_key)])
def remove_node(node: Node):
    """Show available models."""
    try:
        node_url = node.url
        node_manager.remove(node_url)
        logger.info(f'delete node {node_url} successfully')
        return 'Deleted successfully'
    except:  # noqa
        logger.error(f'delete node {node.url} failed.')
        return 'Failed to delete, please check the input url.'


@app.post('/nodes/terminate', dependencies=[Depends(check_api_key)])
def terminate_node(node: Node):
    """Terminate nodes."""
    try:
        node_url = node.url
        success = node_manager.terminate_node(node_url)
        if not success:
            return f'Failed to terminate node {node_url}'
        return 'Terminated successfully'
    except:  # noqa
        logger.error(f'Terminate node {node_url} failed.')
        return 'Failed to terminate node {node_url}, please check the input url.'


@app.get('/nodes/terminate_all', dependencies=[Depends(check_api_key)])
def terminate_node_all():
    """Terminate nodes."""
    try:
        success = node_manager.terminate_all_nodes()
        if not success:
            return 'Failed to terminate all nodes'
        return 'All nodes terminated successfully'
    except:  # noqa
        logger.error('Failed to terminate all nodes')
        return 'Failed to terminate all nodes.'


@app.post('/distserve/connection_warmup')
async def connection_warmup():
    await asyncio.gather(*[
        node_manager.pd_connection_pool.connect(
            PDConnectionMessage(
                p_url=p_url,
                d_url=d_url,
                protocol=node_manager.migration_protocol,
                rdma_config=node_manager.rdma_config,
            )) for p_url in node_manager.prefill_nodes for d_url in node_manager.decode_nodes
    ])
    return JSONResponse({'SUCCESS': True})


@app.post('/distserve/gc')
async def cache_block_gc_to_be_migrated():
    # TODO (JimyMa): add garbage collection of to be migrated request
    raise NotImplementedError


@app.post('/v1/chat/completions', dependencies=[Depends(check_api_key)])
async def chat_completions_v1(request: ChatCompletionRequest, raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Refer to  `https://platform.openai.com/docs/api-reference/chat/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model: model name. Available from /v1/models.
    - messages: string prompt or chat history in OpenAI format. Chat history
        example: `[{"role": "user", "content": "hi"}]`.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - max_tokens (int | None): output token nums. Default to None.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.
    - response_format (Dict | None): Only pytorch backend support formatting
        response. Examples: `{"type": "json_schema", "json_schema": {"name":
        "test","schema": {"properties": {"name": {"type": "string"}},
        "required": ["name"], "type": "object"}}}`
        or `{"type": "regex_schema", "regex_schema": "call me [A-Za-z]{1,10}"}`
    - logit_bias (Dict): Bias to logits. Only supported in pytorch engine.
    - tools (List): A list of tools the model may call. Currently, only
        internlm2 functions are supported as a tool. Use this to specify a
        list of functions for which the model can generate JSON inputs.
    - tool_choice (str | object): Controls which (if any) tool is called by
        the model. `none` means the model will not call any tool and instead
        generates a message. Specifying a particular tool via {"type":
        "function", "function": {"name": "my_function"}} forces the model to
        call that tool. `auto` or `required` will put all the tools information
        to the model.

    Additional arguments supported by LMDeploy:
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
    - min_new_tokens (int): To generate at least numbers of tokens.
    - min_p (float): Minimum token probability, which will be scaled by the
        probability of the most likely token. It must be a value between
        0 and 1. Typical values are in the 0.01-0.2 range, comparably
        selective as setting `top_p` in the 0.99-0.8 range (use the
        opposite of normal `top_p` values)

    Currently we do not support the following features:
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    check_response = await node_manager.check_request_model(request.model)
    if check_response is not None:
        return check_response

    if node_manager.serving_strategy == ServingStrategy.Hybrid:
        node_url = node_manager.get_node_url(request.model)
        if not node_url:
            return node_manager.handle_unavailable_model(request.model)

        logger.info(f'A request is dispatched to {node_url}')
        request_dict = request.model_dump()
        start = node_manager.pre_call(node_url)
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, node_url, '/v1/chat/completions')
            background_task = node_manager.create_background_tasks(node_url, start)
            return StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, node_url, '/v1/chat/completions')
            node_manager.post_call(node_url, start)
            return JSONResponse(json.loads(response))
    elif node_manager.serving_strategy == ServingStrategy.DistServe:
        request_dict = request.model_dump()

        # Prefill
        prefill_request_dict = copy.deepcopy(request_dict)
        prefill_request_dict['max_tokens'] = 1
        prefill_request_dict['stream'] = False
        prefill_request_dict['with_cache'] = True
        prefill_request_dict['preserve_cache'] = True

        prefill_info = {}
        p_url = 'dummy:dummy'
        if not node_manager.dummy_prefill:
            p_url = node_manager.get_node_url(request.model, EngineRole.Prefill)
            if not p_url:
                return node_manager.handle_unavailable_model(request.model)
            logger.info(f'A Prefill request is dispatched to {p_url}')

            start = node_manager.pre_call(p_url)
            prefill_info = json.loads(await node_manager.generate(prefill_request_dict, p_url, '/v1/chat/completions'))
            node_manager.post_call(p_url, start)

        # # Decode
        d_url = node_manager.get_node_url(request.model, EngineRole.Decode, request.input_ids[0])
        if not d_url:
            return node_manager.handle_unavailable_model(request.model)
        logger.info(f'A Decode request is dispatched to {d_url}')

        if not node_manager.dummy_prefill:
            if not node_manager.pd_connection_pool.is_connected(p_url, d_url):
                await node_manager.pd_connection_pool.connect(
                    PDConnectionMessage(
                        p_url=p_url,
                        d_url=d_url,
                        protocol=node_manager.migration_protocol,
                        rdma_config=node_manager.rdma_config,
                    ))

        remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
        remote_block_ids = prefill_info.get('cache_block_ids') or []
        remote_token_id = prefill_info.get('remote_token_ids')[-1] if prefill_info.get('remote_token_ids') else 0

        request_dict['migration_request'] = MigrationRequest(
            protocol=node_manager.migration_protocol,
            remote_engine_id=p_url,
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=node_manager.dummy_prefill).model_dump(mode='json')

        start = node_manager.pre_call(d_url)
        node_manager.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info['id'])
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, d_url, '/v1/chat/completions')
            background_task = node_manager.create_background_tasks(d_url, start)
            resp = StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, d_url, '/v1/chat/completions')
            node_manager.post_call(d_url, start)
            resp = JSONResponse(json.loads(response))

        if not node_manager.dummy_prefill:
            node_manager.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info['id'])

        return resp

    else:
        raise ValueError(f'No serving strategy named {node_manager.serving_strategy}')


@app.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(request: CompletionRequest, raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Go to `https://platform.openai.com/docs/api-reference/completions/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model (str): model name. Available from /v1/models.
    - prompt (str): the input prompt.
    - suffix (str): The suffix that comes after a completion of inserted text.
    - max_tokens (int): output token nums. Default to 16.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - user (str): A unique identifier representing your end-user.
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.

    Additional arguments supported by LMDeploy:
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering

    Currently we do not support the following features:
    - logprobs (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    check_response = await node_manager.check_request_model(request.model)
    if check_response is not None:
        return check_response
    if node_manager.serving_strategy == ServingStrategy.Hybrid:
        node_url = node_manager.get_node_url(request.model)
        if not node_url:
            return node_manager.handle_unavailable_model(request.model)

        logger.info(f'A request is dispatched to {node_url}')
        request_dict = request.model_dump()
        start = node_manager.pre_call(node_url)
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, node_url, '/v1/completions')
            background_task = node_manager.create_background_tasks(node_url, start)
            return StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, node_url, '/v1/completions')
            node_manager.post_call(node_url, start)
            return JSONResponse(json.loads(response))
    elif node_manager.serving_strategy == ServingStrategy.DistServe:
        request_dict = request.model_dump()

        proxy_recv_time = time.time()

        # Prefill
        prefill_request_dict = copy.deepcopy(request_dict)
        prefill_request_dict['max_tokens'] = 1
        prefill_request_dict['stream'] = False
        prefill_request_dict['with_cache'] = True
        prefill_request_dict['preserve_cache'] = True

        if not node_manager.dummy_prefill:
            try:
                p_url = node_manager.get_node_url(request.model, EngineRole.Prefill)
            except Exception as e:
                logger.error(f'error Msg: {str(e)}')
                return {'status': 'Instance sch error, cannot find available p_url'}

            if not p_url:
                return node_manager.handle_unavailable_model(request.model)
            logger.info(f'A Prefill request is dispatched to {p_url}')

            start = node_manager.pre_call(p_url)
            prefill_info = json.loads(await node_manager.generate(prefill_request_dict, p_url, '/v1/completions'))
            node_manager.post_call(p_url, start)
        else:
            p_url = 'dummy:dummy'
            prefill_info = {}

        # Decode
        try:
            d_url = node_manager.get_node_url(request.model, EngineRole.Decode, request.input_ids[0])
        except Exception as e:
            logger.error(f'error Msg: {str(e)}')
            return {'status': 'Instance sch error, cannot find available p_url'}

        if not d_url:
            return node_manager.handle_unavailable_model(request.model)
        logger.info(f'A Decode request is dispatched to {d_url}')

        if not node_manager.dummy_prefill:
            if not node_manager.pd_connection_pool.is_connected(p_url, d_url):
                try:
                    await node_manager.pd_connection_pool.connect(
                        PDConnectionMessage(
                            p_url=p_url,
                            d_url=d_url,
                            protocol=node_manager.migration_protocol,
                            rdma_config=node_manager.rdma_config,
                        ))
                except Exception as e:
                    logger.error(f'error Msg: {str(e)}')
                    return {'status': f'Connection error, cannot establish connection {(p_url, d_url)}'}
            node_manager.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info['id'])

        remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
        remote_block_ids = prefill_info.get('cache_block_ids') or []
        remote_token_id = prefill_info.get('remote_token_ids')[-1] if prefill_info.get('remote_token_ids') else 0
        request_dict['migration_request'] = MigrationRequest(
            protocol=node_manager.migration_protocol,
            remote_engine_id=p_url,
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=node_manager.dummy_prefill).model_dump(mode='json')

        start = node_manager.pre_call(d_url)
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, d_url, '/v1/completions', proxy_recv_time)
            background_task = node_manager.create_background_tasks(d_url, start)
            resp = StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, d_url, '/v1/completions')
            node_manager.post_call(d_url, start)
            node_manager.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info.get('id'))
            resp = JSONResponse(json.loads(response))
        if not node_manager.dummy_prefill:
            node_manager.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info.get('id'))
        return resp
    else:
        raise ValueError(f'No serving strategy named {node_manager.serving_strategy}')


def proxy(server_name: str = '0.0.0.0',
          server_port: int = 8000,
          serving_strategy: Literal['Hybrid', 'DistServe'] = 'Hybrid',
          routing_strategy: Literal['random', 'min_expected_latency', 'min_observed_latency'] = 'min_expected_latency',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False,
          log_level: str = 'INFO',
          disable_cache_status: bool = False,
          link_type: Literal['RoCE', 'IB'] = 'RoCE',
          migration_protocol: Literal['RDMA'] = 'RDMA',
          dummy_prefill: bool = False,
          **kwargs):
    """To launch the proxy server.

    Args:
        server_name (str): the server name of the proxy. Default to '0.0.0.0'.
        server_port (str): the server port. Default to 8000.
        serving_strategy ('Hybrid' | 'DistServe'):  the strategy to serving. Hybrid default.
            DistServe for PD Disaggregation.
        route_strategy ('random' | 'min_expected_latency' | 'min_observed_latency'):
            the strategy to dispatch requests to nodes. Default to
            'min_expected_latency'
        api_keys (List[str] | str | None): Optional list of API keys. Accepts string type as
            a single api_key. Default to None, which means no api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.
        log_level (str): Set the log level. Default to INFO.
        disable_cache_status (str): Whether to cache the proxy status to
             proxy_config.yml.
        migration_protocol: migration protocol when PD disaggregation. RDMA default.
    """  # noqa
    global node_manager
    node_manager = NodeManager(serving_strategy=serving_strategy,
                               routing_strategy=routing_strategy)
    node_manager.migration_protocol = MigrationProtocol[migration_protocol]
    node_manager.dummy_prefill = dummy_prefill

    node_manager.rdma_config = DistServeRDMAConfig(
        link_type=RDMALinkType[link_type],
        with_gdr=True,
    )
    node_manager.cache_status = not disable_cache_status
    if api_keys is not None:
        if isinstance(api_keys, str):
            api_keys = api_keys.split(',')
        from lmdeploy.serve.openai.api_server import VariableInterface
        VariableInterface.api_keys = api_keys
    ssl_keyfile, ssl_certfile = None, None
    if ssl:
        ssl_keyfile = os.environ['SSL_KEYFILE']
        ssl_certfile = os.environ['SSL_CERTFILE']
    logger.setLevel(log_level)
    uvicorn_log_level = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(app=app,
                host=server_name,
                port=server_port,
                log_level=uvicorn_log_level,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile)


if __name__ == '__main__':
    import fire

    fire.Fire(proxy)
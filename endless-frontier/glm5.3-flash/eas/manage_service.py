#!/usr/bin/env python3
"""EAS 服务管理：创建 / 查看 / 日志 / 删除（GLM-5.3-Flash 单机 8 卡）。

用法：
    python3 manage_service.py create --config /path/to/service.json          # 只渲染，不提交
    python3 manage_service.py create --config /path/to/service.json --apply  # 真正创建
    python3 manage_service.py status --name ef_glm53_flash_1node
    python3 manage_service.py logs   --name ef_glm53_flash_1node
    python3 manage_service.py delete --name ef_glm53_flash_1node --apply     # 显式确认才删

约定：
- 真实配置（含 quota、VPC、CPFS id）放在仓库外，不要提交进仓库；仓库里只有 *.template.json。
- 这个服务占 8 张卡（Lingjun 额度），是花钱的资源。验证完请 delete 或把实例数降到 0，
  不要让它空转。
"""

import argparse
import json
import os
import sys
from pathlib import Path

from alibabacloud_eas20210701 import models as eas
from alibabacloud_eas20210701.client import Client as Eas
from alibabacloud_tea_openapi import models as om

ENV = Path(os.path.expanduser("~/.aliyun-ef.env"))
REGION = "cn-shanghai"


def client() -> Eas:
    creds = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return Eas(om.Config(
        access_key_id=creds["ALIBABA_CLOUD_ACCESS_KEY_ID"],
        access_key_secret=creds["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
        endpoint=f"eas.{REGION}.aliyuncs.com",
        region_id=REGION,
        connect_timeout=20000,
        read_timeout=120000,
    ))


def find(eas_client: Eas, name: str) -> dict:
    services = eas_client.list_services(
        eas.ListServicesRequest(page_size=100, page_number=1)
    ).body.to_map().get("Services") or []
    for service in services:
        if service.get("ServiceName") == name:
            return service
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "status", "logs", "delete"))
    parser.add_argument("--config")
    parser.add_argument("--name", default="ef_glm53_flash_1node")
    parser.add_argument("--apply", action="store_true", help="没有它只做只读或渲染")
    parser.add_argument("--lines", type=int, default=120)
    args = parser.parse_args()

    eas_client = client()

    if args.action == "create":
        if not args.config:
            sys.exit("create 需要 --config")
        body = json.loads(Path(args.config).read_text())
        print("将创建的服务：")
        print(json.dumps(body.get("metadata"), ensure_ascii=False, indent=1))
        print("镜像：", body["containers"][0]["image"])
        if not args.apply:
            print("\n（未提交：加 --apply 才会真正创建）")
            return 0
        response = eas_client.create_service(
            eas.CreateServiceRequest(
                develop=False,
                workspace_id=body["metadata"]["workspace_id"],
                body=body,
            )
        ).body.to_map()
        print("已创建：", response.get("ServiceName"), response.get("Status"))
        print("公网入口：", response.get("InternetEndpoint"))
        print("内网入口：", response.get("IntranetEndpoint"))
        return 0

    service = find(eas_client, args.name)
    if not service:
        print("没有找到服务：", args.name)
        return 1

    if args.action == "status":
        for key in ("ServiceName", "Status", "TotalInstance", "RunningInstance",
                    "PendingInstance", "Message", "InternetEndpoint", "IntranetEndpoint",
                    "Gpu", "Cpu", "Memory", "QuotaId", "UpdateTime"):
            print(f"{key}: {service.get(key)}")
        return 0

    if args.action == "logs":
        logs = eas_client.describe_service_log(
            eas.DescribeServiceLogRequest(service_name=args.name, lines=args.lines)
        ).body.to_map()
        print((logs.get("Content") or "")[-6000:])
        return 0

    if args.action == "delete":
        if not args.apply:
            print(f"（未删除：这会停掉 {args.name}；加 --apply 确认）")
            return 0
        eas_client.delete_service(args.name)
        print("已提交删除：", args.name)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

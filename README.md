---
title: "Antani"
author: Giovanni Marelli
date: 2019-11-18
rights:  Creative Commons Non-Commercial Share Alike 3.0
language: en-US
output: 
	md_document:
		variant: markdown_strict+backtick_code_blocks+autolink_bare_uris+markdown_github
---

# antani

Ant - agent/network intelligence 

![antani_logo](docs/f_ops/antani_logo.svg "antani logo")

_ants optimizing paths on a network_

Antani is an agent/network based optimization engine for field operations


# Content

## frontend

We provide a [frontend solution](http://dauvi.org/antani_viz/) under the production vpn

![frontend](f_ops/antani_frontend.png "antani frontend")

And a video explaining the [functioning of the frontend](http://10.0.49.178/antani_demo.mp4)

## infra

[infrastructural design](docs/antani_infra.md) 

* backend - endpoints
* frontend
* aws/productization

![design](docs/f_ops/engine_design.svg "engine design")

_infra design_

## getting started 

### installation

Install the relevant libraries with python>=3.6

```
pip3 install -r requirements.txt
```
### docker 

Build the docker container from the Docker file

```
docker build .
```

### start

Start the service 

```
./script/start_server.sh
```

### apache

Eventually activate apache and do a proxy on the port to avoid CORS restrictions

```
./script/apache.sh
```

configuration files in:

```
./conf/antani_apache.conf
./conf/antani-apache-le-ssl.conf
./onc/enable_proxy.sh
```
## testing

test scripts are in the folder

```
tests/
```

### examples

In the folder `examples` there are few applications to run the library independently from the backend

### pre-trained model

Pre trained models are in the `train` folder.

## kpi

[kpi comparison](docs/antani_kpi.md)

* definition of kpis
* different kpi per run

![kpi](docs/f_ops/kpi_comparison.png "kpi comparison")

_kpi comparison_

## engine

[engine functionalities](docs/mallink_engine.md) 

* list of moves
* performances

![engine](docs/f_ops/vid_phantom.gif "engine")

_engine description_

## graph

[graph building utilities](docs/geomadi_graph.md)

* retrieving a network
* building and fixing a graph

![graph](docs/f_ops/graph_detail.png "graph detail")

_graph formation_

## concept

[basic concepts](docs/antani_concept.md)

* agent
* network optimization

![antani_concept](docs/f_ops/antani_concept.svg "antani concept")

_antani concept schema_


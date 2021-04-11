---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* B.A. Computer Science; Minor in Science and Technology Studies (STS), Cornell University, 2021 (expected)

Work experience
======
* Summer 2021: SWE (expected)
  * Google
  * Duties included: N/A

* Spring 2021: Research Assistant with [CUCAPRA](https://capra.cs.cornell.edu/).
  * Cornell University
  * Duties included: Development of a Number Theoretic Transform (NTT) pipeline generator for Calyx. Expansion of the fixed point library. Lowering a neural network (VGG Net) to Calyx through the TVM Relay frontend.
  * Supervisor: Professor Adrian Sampson

* Fall 2020: Research Assistant with [CUCAPRA](https://capra.cs.cornell.edu/).
  * Cornell University
  * Duties included: Continued development of a TVM Relay frontend for the Calyx language.
  * Supervisor: Professor Adrian Sampson

* Summer 2020: SWE Intern
  * Google Ads
  * Duties included: Improve the de-duplication stage of the crediting pipeline by migrating from two stores to one. It now uses a simpler synchronous callback format while still maintaining the necessary query-per-second rate.
  * Supervisor: Insoo Choo

* Summer 2019: Engineering Practicum
  * Google Cloud
  * Duties included: Optimize the core database implementation of Sawmill Logs, an exabyte scale data lake that supports internal Google analytics. Specifically, I experimented with Zstandard dictionary compression applied across the Google logs database, and compared it to current compression methods.
  * Supervisor: Yixin Luo
* Spring 2019: TA, CS2110 Object-Oriented Programming and Data Structures
  * Cornell University
  * Duties inclued: Facilitated weekly recitation for 40 students, held office hours to assist students in the course, proctored (and graded) examinations and assignments.
* Fall 2018 - Spring 2019: Aerial Robotics at Cornell (ARC)
  * Cornell University
  * Duties included: Improving the proportional-integral-derivative (PID) controller for the quadcopter.
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Member of the Cornell Undergraduate Veteran Association (CUVA). CUVA engages with faculty, staff, and non-veteran students to increase awareness of issues and challenges faced by undergraduate veterans attending Cornell University, both on-campus and off-campus.
* Small unit leader in the Marine Corps. 
* Co-captain of the 1st Battalion, 12th Marines shooting team at the Pacific Division Matches.

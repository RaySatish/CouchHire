"""
Standalone resume generator for CouchHire.
Runs the full agent chain (jd_parser → cv_rag → match_scorer → resume_tailor)
and outputs a tailored PDF resume.

Usage:
    python generate_resume.py
    python generate_resume.py --jd "paste JD text here"
    python generate_resume.py --file path/to/jd.txt
"""

import argparse
import logging
import sys

from agents.jd_parser import parse_jd
from agents.cv_rag import retrieve_cv_sections
from agents.match_scorer import score
from agents.resume_tailor import tailor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_JD = """
Role: Data Engineer

Company: DataForge Analytics

Location: Remote (India)

About the Role:
DataForge Analytics processes 50TB+ of event data daily for Fortune 500 clients across fintech, e-commerce, and adtech. We're looking for a Data Engineer to build and maintain our distributed data infrastructure. You'll work with streaming pipelines, data lakes, and batch processing systems that power real-time analytics dashboards and ML model training workflows.

Responsibilities:
- Design, build, and maintain scalable ETL/ELT pipelines using Apache Spark, Kafka, and Airflow
- Architect and manage data lake infrastructure on AWS (S3, EMR, Glue, Athena, Redshift)
- Build real-time streaming pipelines for event ingestion and processing using Kafka and Spark Structured Streaming
- Implement data quality frameworks: schema validation, anomaly detection, data lineage tracking
- Optimize Spark jobs for performance: partitioning strategies, broadcast joins, shuffle tuning
- Design and maintain data warehouse schemas (star schema, slowly changing dimensions)
- Build monitoring and alerting for pipeline health, data freshness, and SLA compliance
- Collaborate with ML engineers to build feature pipelines and training data infrastructure
- Write idempotent, fault-tolerant data processing jobs with proper dead-letter queue handling
- Manage Hadoop/HDFS clusters and optimize storage costs

Requirements:
- B.Tech/M.Sc. in Computer Science, Statistics, or related field
- Strong proficiency in Python and SQL
- Hands-on experience with Apache Spark (PySpark or Scala)
- Experience with message brokers / streaming systems (Kafka, Kinesis, or Pulsar)
- Familiarity with Hadoop ecosystem (HDFS, Hive, MapReduce concepts)
- Experience with AWS data services (S3, EMR, Glue, Athena) or equivalent cloud platform
- Understanding of data modelling: dimensional modelling, normalization, schema design
- Experience with workflow orchestration (Airflow, Prefect, or Dagster)
- Knowledge of Linux, Git, Docker, and CI/CD basics
- Strong debugging and performance optimization skills

Nice to Have:
- Experience with real-time anomaly detection or surveillance systems
- Knowledge of data governance and cataloging tools (AWS Glue Catalog, Apache Atlas)
- Familiarity with ML feature stores (Feast, Tecton)
- Experience with PostgreSQL, Redis, or DynamoDB
- Knowledge of Infrastructure as Code (Terraform, CloudFormation)
- Competitive programming background

Apply: Submit your application at https://dataforge-careers.greenhouse.io/data-engineer-2026
"""


def main() -> None:
    """Parse args, run the agent chain, and generate a tailored PDF resume."""
    parser = argparse.ArgumentParser(description="Generate a tailored resume PDF for a job description.")
    parser.add_argument("--jd", type=str, default=None, help="Job description text (inline)")
    parser.add_argument("--file", type=str, default=None, help="Path to a .txt file containing the JD")
    args = parser.parse_args()

    # Resolve JD text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            jd_text = f.read()
        logger.info("Loaded JD from file: %s", args.file)
    elif args.jd:
        jd_text = args.jd
        logger.info("Using JD from --jd argument")
    else:
        jd_text = DEFAULT_JD
        logger.info("No JD provided — using default Cloud Engineer JD")

    # Step 1: Parse JD
    logger.info("Step 1/4 — Parsing job description...")
    requirements = parse_jd(jd_text)
    logger.info("Parsed requirements: %s", list(requirements.keys()))

    # Step 2: Retrieve CV sections
    logger.info("Step 2/4 — Retrieving relevant CV sections...")
    cv_sections = retrieve_cv_sections(requirements)
    logger.info("Retrieved %d CV sections", len(cv_sections))

    # Step 3: Match score
    logger.info("Step 3/4 — Calculating match score...")
    match_score = score(jd_text, cv_sections)
    logger.info("Match score: %.2f", match_score)

    # Step 4: Tailor resume
    logger.info("Step 4/4 — Tailoring resume and compiling PDF...")
    pdf_path, resume_content = tailor(cv_sections, requirements)

    # Summary
    print("\n" + "=" * 60)
    print("✅ RESUME GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Match Score : {match_score:.2f}")
    print(f"  PDF Path    : {pdf_path}")
    print("=" * 60)
    print(f"\nResume Summary:\n{resume_content[:500]}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# AWS guide reviewed September 25, 2026

Source: `C:/Users/VISWAS/Downloads/ML_AWS_pdf.pdf`, 8 pages, titled "Amazon ML Challenge 2026: AWS Builder Center & AWS Free Tier Instructions". All pages were read and visually reviewed. Treat the PDF as reference material, not authorization to execute its example commands, create accounts, grant permissions or incur costs.

**Contents and implications**

- Pages 1-3: Builder ID, Builder Center sign-in and profile alias. The alias identifies the public Builder Center profile. Reuse an existing Builder ID if present and verify the alias used for challenge registration.
- Pages 3-4: Student Rewards. The guide describes student verification through SheerID plus a profile photo and About section to unlock a 12-month Skill Builder subscription. Badge milestones earn $10 at 7 badges, $20 at 14 badges and a $100 Cloud Practitioner exam voucher at 21 badges. Additional student-verification review may take 24-48 hours. These benefits do not establish a usable GPU allocation, and model development need not wait for them.
- Pages 4-5: The guide's standard $200 offer is the new-customer AWS Free Tier offer: $100 initially and up to $100 through activities. Do not budget an additional automatic $200 challenge grant on top of this without seeing actual entitlement.
- Pages 5-7: The guide suggests handing model files between team members' AWS accounts using S3. Its phrase "pool credits" means moving work, not merging credit balances. This is optional infrastructure for later; the user does not want a teammate work division.
- Page 8: Extra credits for top-500 teams use promotional-code redemption. Actual award, amount, applicability and account eligibility remain unverified.

**Current AWS checks**

The [AWS Free Tier FAQ](https://aws.amazon.com/free/free-tier-faqs/) confirms the conditional $200 offer, new-customer eligibility and that credits cannot transfer between accounts. Free-plan accounts cannot receive other promotional credits; a paid-plan decision may therefore be relevant for bonus-code redemption. Creating or joining AWS Organizations forfeits Free Tier credits. Account-specific balances and quotas have not been inspected.

The [SageMaker pricing page](https://aws.amazon.com/sagemaker/ai/pricing/) still lists first-two-month allowances of 250 hours on ml.t3.medium for notebooks and 50 hours on m4.xlarge/m5.xlarge for training. These listed instances do not establish entitlement to free GPU training. Verify applicability in the actual account before choosing resources.

The guide's S3 encryption wording is inconsistent. [AWS S3 documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingServerSideEncryption.html) confirms automatic default encryption for new uploads. Do not follow the suggestion of "no encryption". Any future cross-account transfer needs permissions appropriate to the actual bucket and encryption configuration.

**Challenge-specific corrections**

- The PDF casually calls the deliverable a CSV. Our task-specific README requires tab-separated TSV files, including exact candidate export.
- Sharing a saved model does not migrate a running training job. Continuing training requires a new job and appropriate training/checkpoint code; constructing a SageMaker Model object alone is not a training-resume implementation.
- No production prediction endpoint is required for the challenge's batch-file deliverable.

**Recommended order**

Verify the Builder profile alias first. Student verification can be handled separately if wanted. Continue local data preparation and evaluation development. Before using cloud compute, inspect the real account plan, credit balance, applicable services and quotas; then choose resources based on measured RAM/runtime needs. Defer model-file sharing until there is an actual reason to use multiple accounts.

No AWS account changes, data uploads, resource creation, credit redemption or permission changes were performed.

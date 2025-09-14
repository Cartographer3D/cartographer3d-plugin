// Load configuration
const config = require("../auto-label-config.json");

/**
 * Determines labels based on PR title
 * @param {string} title - The PR title
 * @returns {string[]} Array of label names
 */
function getLabelsFromTitle(title) {
  const normalizedTitle = title.toLowerCase();
  const labels = [];

  // Check for type-based labels
  for (const [prefix, label] of Object.entries(config.titleMappings)) {
    if (normalizedTitle.startsWith(prefix)) {
      labels.push(label);
      break;
    }
  }

  // Check for breaking change
  if (normalizedTitle.includes(config.breakingChangeIndicator)) {
    labels.push("breaking change");
  }

  return labels;
}

/**
 * Checks if a chore PR should be ignored for release
 * @param {Object} github - GitHub API client
 * @param {Object} context - GitHub context
 * @param {string[]} labels - Current labels for the PR
 * @returns {Promise<boolean>} Whether to add ignore-for-release label
 */
async function shouldIgnoreForRelease(github, context, labels) {
  // Only apply to chore PRs
  if (!labels.includes("chore")) {
    return false;
  }

  try {
    // Get the list of changed files in this PR
    const files = await github.rest.pulls.listFiles({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: context.payload.pull_request.number,
    });

    // Convert string patterns to regex
    const releaseRelevantPatterns = config.releaseRelevantPaths.map(
      (pattern) => new RegExp(pattern),
    );

    // Check if any changed file matches release-relevant paths
    const touchesReleaseRelevantFiles = files.data.some((file) =>
      releaseRelevantPatterns.some((pattern) => pattern.test(file.filename)),
    );

    // If chore doesn't touch release-relevant files, mark as ignore-for-release
    return !touchesReleaseRelevantFiles;
  } catch (error) {
    console.log("Error checking files for release relevance:", error);
    return false;
  }
}

/**
 * Calculates which labels to add and remove
 * @param {string[]} existingLabels - Currently applied labels
 * @param {string[]} targetLabels - Labels that should be applied
 * @returns {Object} Object with labelsToAdd and labelsToRemove arrays
 */
function calculateLabelChanges(existingLabels, targetLabels) {
  // Only consider managed labels for removal
  const labelsToRemove = existingLabels.filter(
    (label) =>
      config.managedLabels.includes(label) && !targetLabels.includes(label),
  );

  // Only add labels that aren't already present
  const labelsToAdd = targetLabels.filter(
    (label) => !existingLabels.includes(label),
  );

  return { labelsToAdd, labelsToRemove };
}

/**
 * Removes labels from a PR
 * @param {Object} github - GitHub API client
 * @param {Object} context - GitHub context
 * @param {string[]} labelsToRemove - Labels to remove
 */
async function removeLabels(github, context, labelsToRemove) {
  for (const label of labelsToRemove) {
    try {
      await github.rest.issues.removeLabel({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.payload.pull_request.number,
        name: label,
      });
      console.log(`Removed label: ${label}`);
    } catch (error) {
      console.log(`Error removing label ${label}:`, error);
    }
  }
}

/**
 * Adds labels to a PR
 * @param {Object} github - GitHub API client
 * @param {Object} context - GitHub context
 * @param {string[]} labelsToAdd - Labels to add
 */
async function addLabels(github, context, labelsToAdd) {
  if (labelsToAdd.length === 0) {
    return;
  }

  try {
    await github.rest.issues.addLabels({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.payload.pull_request.number,
      labels: labelsToAdd,
    });
    console.log(`Added labels: ${labelsToAdd.join(", ")}`);
  } catch (error) {
    console.log("Error adding labels:", error);
  }
}

/**
 * Main function to auto-label a PR
 * @param {Object} params - Object containing github, context, and console
 */
module.exports = async ({ github, context, console: logger }) => {
  const title = context.payload.pull_request.title;
  const existingLabels = context.payload.pull_request.labels.map(
    (label) => label.name,
  );

  logger.log(`Processing PR: "${title}"`);
  logger.log(`Existing labels: ${existingLabels.join(", ") || "none"}`);

  // Determine labels based on title
  const titleBasedLabels = getLabelsFromTitle(title);
  logger.log(`Title-based labels: ${titleBasedLabels.join(", ") || "none"}`);

  // Check if we should add ignore-for-release label
  const shouldIgnore = await shouldIgnoreForRelease(
    github,
    context,
    titleBasedLabels,
  );

  // Build final target labels
  const targetLabels = [...titleBasedLabels];
  if (shouldIgnore) {
    targetLabels.push(config.ignoreForReleaseLabel);
  }

  logger.log(`Target labels: ${targetLabels.join(", ") || "none"}`);

  // Calculate what changes are needed
  const { labelsToAdd, labelsToRemove } = calculateLabelChanges(
    existingLabels,
    targetLabels,
  );

  if (labelsToRemove.length === 0 && labelsToAdd.length === 0) {
    logger.log("No label changes needed");
    return;
  }

  logger.log(`Labels to remove: ${labelsToRemove.join(", ") || "none"}`);
  logger.log(`Labels to add: ${labelsToAdd.join(", ") || "none"}`);

  // Apply changes
  await removeLabels(github, context, labelsToRemove);
  await addLabels(github, context, labelsToAdd);

  logger.log("Label update complete");
};

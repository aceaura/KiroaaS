// Usage-query script for cc-switch, reading this gateway's GET /usage endpoint.
//
// Paste the whole file into the provider's "usage script" field. cc-switch
// substitutes {{baseUrl}} and {{apiKey}} from that provider's own settings
// (ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN), so the same text works for every
// instance and no key is hard-coded here.
//
// /usage returns the upstream GetUsageLimits payload verbatim. /account is the
// friendlier shape but reports resetDate as null — it reads a field name the
// upstream response does not use — so the raw endpoint is the reliable source.
({
  request: {
    // Trailing slashes on the configured base URL would otherwise produce
    // "//usage", which the gateway answers with 404.
    url: "{{baseUrl}}".replace(/\/+$/, "") + "/usage",
    method: "GET",
    headers: {
      "Authorization": "Bearer {{apiKey}}",
      "User-Agent": "cc-switch/1.0"
    }
  },
  extractor: function (response) {
    if (!response || typeof response !== "object") {
      return { isValid: false, invalidMessage: "KiroaaS 无响应" };
    }
    // FastAPI reports errors as {"detail": "..."}: 401 on a bad key, 503 when no
    // account initialized, or an upstream status passed straight through.
    if (response.detail) {
      return { isValid: false, invalidMessage: "KiroaaS: " + response.detail };
    }

    var list = response.usageBreakdownList;
    if (!list || !list.length) {
      return { isValid: false, invalidMessage: "KiroaaS 未返回用量明细" };
    }
    var item = list[0];

    // The *WithPrecision fields carry fractional credits; the plain ones are
    // truncated. Prefer precision, fall back for older payload shapes.
    var total = item.usageLimitWithPrecision;
    if (total == null) total = item.usageLimit;
    var used = item.currentUsageWithPrecision;
    if (used == null) used = item.currentUsage;
    if (total == null || used == null) {
      return { isValid: false, invalidMessage: "KiroaaS 用量字段缺失" };
    }
    // Rounded because float subtraction yields e.g. 454.14999999999964.
    var remaining = Math.round((total - used) * 100) / 100;

    var sub = response.subscriptionInfo || {};
    var extra = "";

    var reset = response.nextDateReset || item.nextDateReset;
    if (reset) {
      var d = new Date(reset * 1000);
      if (!isNaN(d.getTime())) {
        extra = "重置 " + (d.getMonth() + 1) + "-" + d.getDate();
      }
    }

    var overage = (response.overageConfiguration || {}).overageStatus;
    if (overage === "ENABLED") {
      var cap = item.overageCapWithPrecision;
      if (cap == null) cap = item.overageCap;
      extra += (extra ? " | " : "") + "超额上限 " + (cap == null ? "?" : cap);
    }

    var bonuses = item.bonuses || [];
    var bonusRemaining = 0;
    var hasBonus = false;
    for (var i = 0; i < bonuses.length; i++) {
      var b = bonuses[i];
      if (b.status !== "ACTIVE") continue;
      var bt = b.usageLimitWithPrecision;
      if (bt == null) bt = b.usageLimit || 0;
      var bu = b.currentUsageWithPrecision;
      if (bu == null) bu = b.currentUsage || 0;
      bonusRemaining += bt - bu;
      hasBonus = true;
    }
    if (hasBonus) {
      extra += (extra ? " | " : "") + "奖励余额 " + Math.round(bonusRemaining * 100) / 100;
    }

    return {
      isValid: true,
      planName: sub.subscriptionTitle || sub.type || "Kiro",
      used: used,
      total: total,
      remaining: remaining,
      unit: item.displayNamePlural || item.displayName || "Credits",
      extra: extra
    };
  }
})

"use client";

import { motion } from "framer-motion";
import { Check, ArrowRight } from "lucide-react";
import { useState } from "react";

const PLANS = [
  {
    name: "Free",
    price: { monthly: "$0", yearly: "$0" },
    desc: "For side projects and exploration.",
    features: ["3 active projects", "Core components", "Community support", "Basic analytics"],
    cta: "Get started",
    accent: false,
  },
  {
    name: "Pro",
    price: { monthly: "$18", yearly: "$14" },
    desc: "For serious makers moving fast.",
    features: ["Unlimited projects", "All components", "Priority support", "Advanced analytics", "API access", "Custom domains"],
    cta: "Start free trial",
    accent: true,
  },
  {
    name: "Team",
    price: { monthly: "$49", yearly: "$39" },
    desc: "For teams shipping together.",
    features: ["Everything in Pro", "Up to 20 seats", "SSO / SAML", "Audit logs", "Dedicated Slack", "SLA guarantee"],
    cta: "Talk to sales",
    accent: false,
  },
];

export default function Pricing() {
  const [annual, setAnnual] = useState(false);

  return (
    <section className="bg-black border-t border-white/10 py-32 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-16 gap-6">
          <div>
            <div className="flex items-center gap-3 mb-5">
              <span className="w-6 h-px bg-[#FF2D55]" />
              <span className="text-[#FF2D55] text-xs tracking-[0.25em] uppercase">Pricing</span>
            </div>
            <h2
              className="text-white font-black tracking-tighter"
              style={{ fontSize: "clamp(2rem, 5vw, 4rem)" }}
            >
              Honest pricing.
              <br />
              <span className="text-white/25">No surprises.</span>
            </h2>
          </div>

          {/* Toggle */}
          <div className="flex items-center gap-4">
            <span className={`text-xs tracking-widest uppercase ${!annual ? "text-white" : "text-white/30"}`}>Monthly</span>
            <button
              onClick={() => setAnnual(!annual)}
              className="w-12 h-6 rounded-full relative transition-colors duration-300"
              style={{ background: annual ? "#FF2D55" : "rgba(255,255,255,0.1)" }}
            >
              <motion.div
                animate={{ x: annual ? 24 : 2 }}
                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                className="absolute top-1 w-4 h-4 bg-white rounded-full"
              />
            </button>
            <span className={`text-xs tracking-widest uppercase ${annual ? "text-white" : "text-white/30"}`}>
              Annual <span className="text-[#FF2D55]">–22%</span>
            </span>
          </div>
        </div>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-white/5">
          {PLANS.map(({ name, price, desc, features, cta, accent }, i) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="relative flex flex-col p-8"
              style={{ background: accent ? "rgba(255,45,85,0.05)" : "#000" }}
            >
              {accent && (
                <div className="absolute top-0 left-0 right-0 h-px bg-[#FF2D55]" />
              )}

              <div className="mb-8">
                <div className="flex items-center justify-between mb-4">
                  <span className={`text-xs tracking-widest uppercase ${accent ? "text-[#FF2D55]" : "text-white/30"}`}>
                    {name}
                  </span>
                  {accent && (
                    <span className="text-[10px] bg-[#FF2D55] text-white px-2 py-0.5 rounded-sm font-bold tracking-wider uppercase">
                      Popular
                    </span>
                  )}
                </div>

                <div className="flex items-baseline gap-1 mb-3">
                  <motion.span
                    key={annual ? "annual" : "monthly"}
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-white font-black text-5xl tracking-tighter"
                  >
                    {annual ? price.yearly : price.monthly}
                  </motion.span>
                  <span className="text-white/25 text-sm">/mo</span>
                </div>
                <p className="text-white/35 text-sm">{desc}</p>
              </div>

              <ul className="space-y-3 mb-10 flex-1">
                {features.map((f) => (
                  <li key={f} className="flex items-center gap-3 text-sm text-white/50">
                    <Check size={12} className="text-[#FF2D55] shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
                className="w-full py-3.5 text-xs font-bold tracking-widest uppercase flex items-center justify-center gap-2 rounded-sm transition-colors duration-300"
                style={{
                  background: accent ? "#FF2D55" : "transparent",
                  color: accent ? "#fff" : "rgba(255,255,255,0.5)",
                  border: accent ? "none" : "1px solid rgba(255,255,255,0.08)",
                }}
                onMouseEnter={(e) => {
                  if (!accent) {
                    e.currentTarget.style.color = "#fff";
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.3)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!accent) {
                    e.currentTarget.style.color = "rgba(255,255,255,0.5)";
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                  }
                }}
              >
                {cta} <ArrowRight size={13} />
              </motion.button>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
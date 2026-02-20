"use client";

import { motion } from "framer-motion";
import { Zap, Layers, Shield, Move, Code2, Gauge } from "lucide-react";

const FEATURES = [
  { icon: Zap,     label: "01", title: "Lightning Fast",     desc: "Zero-config performance. Sub-50ms interactions out of the box, no compromises." },
  { icon: Layers,  label: "02", title: "Composable System",  desc: "Primitives that stack. Build complex UIs from a tight set of atomic pieces." },
  { icon: Shield,  label: "03", title: "Enterprise Ready",   desc: "SOC-2. SSO. RBAC. Compliance baked in, not bolted on." },
  { icon: Move,    label: "04", title: "Drag & Customize",   desc: "Every layout, every token — yours to own. No lock-in, ever." },
  { icon: Code2,   label: "05", title: "Developer First",    desc: "Typed APIs, sensible defaults, exhaustive docs. Ship in minutes." },
  { icon: Gauge,   label: "06", title: "Real-time Analytics",desc: "Watch your product live. Heatmaps, funnels, and events — all in one." },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

const item = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.16, 1, 0.3, 1] } },
};

export default function Features() {
  return (
    <section className="bg-black py-32 px-6 border-t border-white/10">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-20 gap-6">
          <div>
            <div className="flex items-center gap-3 mb-5">
              <span className="w-6 h-px bg-[#FF2D55]" />
              <span className="text-[#FF2D55] text-xs tracking-[0.25em] uppercase">Capabilities</span>
            </div>
            <h2
              className="text-white font-black tracking-tighter leading-tight"
              style={{ fontSize: "clamp(2rem, 5vw, 4rem)" }}
            >
              Everything you need.
              <br />
              <span className="text-white/25">Nothing you don't.</span>
            </h2>
          </div>
          <p className="text-white/40 text-sm max-w-xs leading-relaxed md:text-right">
            We ruthlessly cut features that don't serve you. What's left is pure signal.
          </p>
        </div>

        {/* Grid */}
        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-white/5"
        >
          {FEATURES.map(({ icon: Icon, label, title, desc }) => (
            <motion.div
              key={title}
              variants={item}
              className="group bg-black p-8 hover:bg-[#FF2D55]/5 transition-colors duration-500 cursor-default"
            >
              <div className="flex items-start justify-between mb-8">
                <span className="text-white/15 text-xs tracking-widest font-mono">{label}</span>
                <div className="w-9 h-9 border border-white/10 group-hover:border-[#FF2D55]/40 flex items-center justify-center rounded-sm transition-colors duration-300">
                  <Icon size={16} className="text-white/30 group-hover:text-[#FF2D55] transition-colors duration-300" />
                </div>
              </div>
              <h3 className="text-white font-bold text-base mb-3 tracking-tight">{title}</h3>
              <p className="text-white/35 text-sm leading-relaxed">{desc}</p>

              {/* Hover bar */}
              <div className="mt-8 h-px bg-white/5 group-hover:bg-[#FF2D55]/40 transition-all duration-500" />
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
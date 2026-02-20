"use client";

import { motion } from "framer-motion";
import { Quote } from "lucide-react";

const TESTIMONIALS = [
  {
    quote: "Pulse cut our design-to-ship time in half. It's the most focused tool we've ever used.",
    name: "Alex Chen",
    role: "Design Lead · Vercel",
    index: "01",
  },
  {
    quote: "Finally — a product that doesn't try to be everything. The minimalism is the feature.",
    name: "Priya Mehta",
    role: "CTO · Linear",
    index: "02",
  },
  {
    quote: "I switched entire teams onto Pulse within a week. It just works, beautifully.",
    name: "Jake Rivera",
    role: "Founder · Craft",
    index: "03",
  },
];

export default function Testimonials() {
  return (
    <section className="bg-black border-t border-white/10 py-32 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-20">
          <span className="w-6 h-px bg-[#FF2D55]" />
          <span className="text-[#FF2D55] text-xs tracking-[0.25em] uppercase">What people say</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-white/5">
          {TESTIMONIALS.map(({ quote, name, role, index }, i) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="bg-black p-8 flex flex-col justify-between gap-10"
            >
              <div>
                <div className="flex items-start justify-between mb-6">
                  <Quote size={20} className="text-[#FF2D55]/50" />
                  <span className="text-white/10 text-xs font-mono tracking-widest">{index}</span>
                </div>
                <p className="text-white/70 text-lg leading-relaxed font-light">
                  "{quote}"
                </p>
              </div>
              <div className="border-t border-white/5 pt-6">
                <p className="text-white text-sm font-semibold">{name}</p>
                <p className="text-white/30 text-xs mt-1 tracking-wide">{role}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
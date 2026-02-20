"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { useRef } from "react";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.3 } },
};

const word = {
  hidden: { y: "110%", opacity: 0 },
  show: { y: "0%", opacity: 1, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } },
};

export default function Hero() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref });
  const y = useTransform(scrollYProgress, [0, 1], ["0%", "30%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.6], [1, 0]);

  const headline = ["Build", "Faster.", "Ship", "Better."];

  return (
    <section
      ref={ref}
      className="min-h-screen bg-black flex flex-col justify-end pb-20 px-6 pt-28 relative overflow-hidden"
    >
      {/* Background grid */}
      <div
        className="absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage: `linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)`,
          backgroundSize: "80px 80px",
        }}
      />

      {/* Red blur accent */}
      <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-[#FF2D55] rounded-full blur-[160px] opacity-15 pointer-events-none" />

      <motion.div style={{ y, opacity }} className="max-w-7xl mx-auto w-full relative z-10">
        {/* Eyebrow */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex items-center gap-3 mb-12"
        >
          <span className="w-8 h-px bg-[#FF2D55]" />
          <span className="text-[#FF2D55] text-xs tracking-[0.25em] uppercase font-medium">
            Design system for the bold
          </span>
        </motion.div>

        {/* Headline */}
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="overflow-hidden"
        >
          <h1 className="font-black leading-[0.88] tracking-tighter text-white"
            style={{ fontSize: "clamp(3.5rem, 12vw, 11rem)" }}>
            {headline.map((w, i) => (
              <div key={i} className="overflow-hidden inline-block mr-[0.2em]">
                <motion.span variants={word} className="inline-block">
                  {w}
                </motion.span>
              </div>
            ))}
          </h1>
        </motion.div>

        {/* Bottom row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="mt-16 flex flex-col md:flex-row items-start md:items-end justify-between gap-8 border-t border-white/10 pt-8"
        >
          <p className="text-white/40 max-w-xs text-sm leading-relaxed">
            The minimal design toolkit for product teams who care about every pixel. Less noise, more signal.
          </p>

          <div className="flex items-center gap-4">
            <motion.a
              href="#"
              whileHover={{ scale: 1.03, backgroundColor: "#FF2D55" }}
              whileTap={{ scale: 0.96 }}
              className="flex items-center gap-2 px-7 py-3.5 bg-white text-black text-xs font-bold tracking-widest uppercase rounded-sm transition-colors duration-300"
            >
              Start building <ArrowRight size={14} />
            </motion.a>
            <motion.a
              href="#"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.96 }}
              className="flex items-center gap-2 text-white/50 hover:text-white text-xs tracking-widest uppercase transition-colors duration-200"
            >
              View docs <ArrowUpRight size={14} />
            </motion.a>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1.2 }}
          className="mt-16 grid grid-cols-3 gap-px bg-white/5"
        >
          {[["12k+", "Users"], ["99.9%", "Uptime"], ["< 50ms", "Response"]].map(([num, label]) => (
            <div key={label} className="bg-black py-6 px-4 text-center">
              <div className="text-white text-2xl font-black tracking-tight">{num}</div>
              <div className="text-white/30 text-xs tracking-widest uppercase mt-1">{label}</div>
            </div>
          ))}
        </motion.div>
      </motion.div>
    </section>
  );
}
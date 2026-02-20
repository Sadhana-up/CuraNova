"use client";

import { motion } from "framer-motion";
import { Zap, Github, Twitter, Linkedin } from "lucide-react";

const LINKS = {
  Product: ["Features", "Pricing", "Changelog", "Roadmap"],
  Company: ["About", "Blog", "Careers", "Press"],
  Legal: ["Privacy", "Terms", "Cookies", "License"],
};

export default function Footer() {
  return (
    <footer className="bg-black border-t border-white/10 pt-20 pb-10 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Top */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-12 mb-16">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-7 h-7 bg-[#FF2D55] flex items-center justify-center rounded-sm">
                <Zap size={14} className="text-white fill-white" />
              </div>
              <span className="text-white font-bold text-sm tracking-widest uppercase">Pulse</span>
            </div>
            <p className="text-white/30 text-xs leading-relaxed max-w-[180px]">
              The minimal design system built for speed and clarity.
            </p>
            <div className="flex items-center gap-4 mt-6">
              {[Github, Twitter, Linkedin].map((Icon, i) => (
                <motion.a
                  key={i}
                  href="#"
                  whileHover={{ scale: 1.15, color: "#FF2D55" }}
                  className="text-white/30 transition-colors duration-200"
                >
                  <Icon size={16} />
                </motion.a>
              ))}
            </div>
          </div>

          {/* Nav columns */}
          {Object.entries(LINKS).map(([group, links]) => (
            <div key={group}>
              <p className="text-white text-xs font-bold tracking-widest uppercase mb-5">{group}</p>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link}>
                    <a
                      href="#"
                      className="text-white/30 hover:text-white text-xs tracking-wide transition-colors duration-200"
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom */}
        <div className="border-t border-white/5 pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-white/20 text-xs tracking-wide">© 2025 Pulse, Inc. All rights reserved.</p>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
            <span className="text-white/20 text-xs">All systems operational</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
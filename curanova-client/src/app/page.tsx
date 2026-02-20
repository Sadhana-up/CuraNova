import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Marquee from "@/components/Marquee";
import Features from "@/components/Features";
import Testimonials from "@/components/Testimonials";
import Pricing from "@/components/Pricing";
import CTA from "@/components/Cta";
import Footer from "@/components/Footer";

export default function LandingPage() {
  return (
    <main className="bg-black min-h-screen">
      <Navbar />
      <Hero />
      <Marquee />
      <Features />
      <Testimonials />
      <Pricing />
      <CTA />
      <Footer />
    </main>
  );
}
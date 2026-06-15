import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pluno Doc Updater",
  description: "Update documentation from natural-language change requests",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** The file name without its book extension. */
export const bookTitle = (filename: string) => filename.replace(/\.(epub|pdf|cbz|zip)$/i, "");

/** "EPUB · 12 chapters", "PDF · 300 pages", "Manga · 74 pages". */
export function bookKind(book: { filename: string; total_chapters: number; manga?: boolean | null }) {
  const isPdf = book.filename.toLowerCase().endsWith(".pdf");
  const kind = book.manga ? "Manga" : isPdf ? "PDF" : "EPUB";
  return `${kind} · ${book.total_chapters} ${book.manga || isPdf ? "pages" : "chapters"}`;
}

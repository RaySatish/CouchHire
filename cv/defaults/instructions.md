# Tailoring Instructions (Default)

> These are generic defaults. Place your own `instructions.md` in `cv/uploads/`
> to override with personal rules.

## General
- Keep resume to 1 page maximum
- Quantify achievements where possible
- Do not include references or hobbies

## Skills — Selection Rules

> Skills are controlled at two levels: **categories** (include/exclude entire categories)
> and **individual items** (include/exclude specific skills within a category).

### How it works
1. **Category-level:** The system selects which skill categories to include based on JD
   relevance and your instructions. Irrelevant categories are excluded entirely.
2. **Item-level:** Within included categories, individual skills can be excluded if
   they are not relevant to the JD or if your instructions say to remove them.
3. **Ordering:** Categories are ordered by relevance to the JD. You can pin specific
   categories to fixed positions (e.g., "Programming always at the bottom").

### Master CV formatting
- Structure skills as one `\textbf{Category:}` per `\item` line
- Each category should list its skills as comma-separated items
- The system parses categories from `\textbf{Name:}` patterns — keep this format consistent

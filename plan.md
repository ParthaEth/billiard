# Web plan

Not built yet. This is the intended shape of a site where any BVBW player can sign in and log club games that feed the same Elo fit.

The +4.30 offset in `scale.cfg` stays fixed. The site publishes fitted Elo plus that offset. Never recompute the offset and never overwrite `data/scale_reference.csv`.

## Hosting

Use three free services. Cloudflare is not part of this.

- **Vercel Hobby** hosts the site and the API. The free plan covers a personal, non-commercial project: about 100 GB transfer, 1 million function calls, and 4 CPU-hours a month. Python functions are allowed (about 500 MB, up to 5 minutes). Vercel does not provide a database. If the site later takes payment, Hobby is no longer the right plan.
- **Neon** is the database: free Postgres, 1 GB per project. It sleeps when idle and wakes on the next request. Supabase is the other free Postgres and includes login, but a free project pauses after a week of no use and has to be switched back on by hand. A league site goes quiet between match nights, so Neon is the better fit.
- **Resend** sends the mail: 3,000 emails a month, 100 a day, on the free plan. Vercel does not send email. Use it for sign-in links, the captain verification key, and opponent confirmation.

The website does not need a paid domain. Vercel gives a free address such as `billiard.vercel.app`, and a custom domain can be attached later without rebuilding.

Mail can go through a personal Gmail account, which also needs no domain. The server is `smtp.gmail.com` on port 587. Google no longer accepts the normal account password: turn on 2-Step Verification and create an app password, and store that only as a secret on the host. Messages are sent from that Gmail address. A personal account can send about 500 messages a day, which is enough for sign-in links and captain keys. Vercel allows this on ports 465 and 587. A second free Gmail, used only by the site, keeps the everyday inbox separate if Google ever pauses the sender.

Resend remains the alternative if Gmail is not used. Its shared sender delivers only to the address on the Resend account, so mail to captains would then need a domain you control. A `.de` or `.com` is about €10–15 a year.

Keep the heavy work off the website. `sync.py` and `elo.py --half-life-years 0.67` stay a Python job on this machine, or later a scheduled GitHub Action. That job writes ratings into Neon. Pages only read them. A joint refit does not run on each page view.

## Accounts

Anyone joins with an email, the same way as signing in. The first time, before the account exists, they pick one official profile: the display name and its Pass-Nr. from the roster. One account per Pass-Nr., and one Pass-Nr. per account.

That claim does not verify itself. The site emails a verification key to the **team captain** published for that player's current squad, not to the player. The player can log club games only after that key has been used and the account is verified. Until then they can sign in and see ratings.

If that captain already has an account, the same key is shown there, signed in as the captain's Pass-Nr. It is not shown on a public page. That covers mail that never arrives. The captain can read the key on the site and pass it to the player. A claim still waits when the captain has neither a published email nor an account.

Open point, already measured: many squads publish no captain email. Tübinger BC captains in 2026/27 have none. Those claims move only once the captain has an account, or an email is published. A later fallback (a verified teammate, or a code shown at the next league match) stays out of scope.

Captain addresses are used only to send that key. They are not shown on the site and not used for anything else.

## Club games

A verified player enters a club game against another roster player. The game is stored immediately and shown as pending. It is confirmed only when the opponent, signed in as that Pass-Nr., accepts it. Only confirmed club games enter the Elo fit, on the next run of `elo.py`, together with the league and Einzel games already in `data/games.csv`.

Either player can dispute a pending game. A disputed game never enters the fit.

Players with no Pass-Nr. stay as they are today (`data/inhouse.csv`, `AGENT_TODO.md`). The site does not invent a Pass-Nr. for them.

## Later, not now

No accounts, no mail, and no host until the pieces above are chosen and a captain-email path is tested on one club that actually publishes one. Opponent vouching, QR codes at the table, and public rating pages for unclaimed players wait until sign-in and confirmed games work.
